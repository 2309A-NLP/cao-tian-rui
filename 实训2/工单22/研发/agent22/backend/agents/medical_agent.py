"""
医疗 Agent — 对接工单12（健康咨询 + Neo4j 知识图谱）

流程：
  recall 历史记忆
    → 拼 query（记忆块 + 本轮问题）
    → POST http://localhost:8012/chat  （工单12 真实 Agent）
    → 若工单12不可用，fallback 到本地 LLM
    → remember（fire-and-forget）
    → 返回 reply + recalled + source

本模块在教育/文旅 Agent 的基础上增加了"优先调用外部服务，失败则本地降级"的模式，
体现了微服务架构中的熔断/降级设计。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# logging：标准库，用于记录结构化日志
import logging
# os：标准库，用于读取环境变量（WT12_URL、WT12_TIMEOUT）
import os
# time：标准库，高精度计时器
import time
# Optional：typing 标准库，声明可为 None 的参数类型
from typing import Optional

# httpx：第三方 HTTP 客户端库（类似 requests），支持同步和异步调用，
# 用于在医疗 Agent 内部向工单12的服务端点发起 HTTP POST 请求
import httpx

# dotenv：第三方包，从 .env 文件加载环境变量到 os.environ
from dotenv import load_dotenv

# 导入本项目 LLM 封装（fallback 路径使用）
from src.llm_client import LLMClient
# 导入记忆客户端单例和 user_id 构造函数
from src.memory_client import MemoryClient, make_user_id

# 重新加载 .env（防止此模块被单独 import 时环境变量未初始化）
load_dotenv()
# 获取当前模块的 logger，日志前缀 "agent22.agent.medical"
logger = logging.getLogger("agent22.agent.medical")

# 工单12 对话接口地址：优先从环境变量读取，默认本地 8012 端口
WT12_URL     = os.getenv("WT12_URL", "http://localhost:8012/chat")
# 调用工单12的超时时间（秒），默认 30 秒（工单12内部有 Neo4j 图查询，耗时可能较长）
WT12_TIMEOUT = float(os.getenv("WT12_TIMEOUT", "30"))

# ── Fallback System Prompt：工单12不可用时使用本地 LLM 的角色定义 ──────────────────────────────────────
# 定义 LLM 在 fallback 场景下的行为准则
FALLBACK_SYSTEM = """你是一名严谨的医疗健康助理。请遵循以下原则：

1. 回答用户健康咨询时，必须先阅读【患者历史记忆】，并在回复中合理利用。
2. 涉及处方/用药建议时，务必核对既往过敏史和当前用药，避免冲突。
3. 使用简体中文回答，语气温和专业。
4. 如无法确定，明确告知患者需线下就诊，不做过度承诺。
5. 每次回复末尾附加免责声明："（本回复仅供参考，不能替代医生面诊）"。
"""

# User Prompt 模板：将记忆块和问题注入后，作为统一格式传给工单12或本地 LLM
USER_TEMPLATE = """【患者历史记忆】
{memory_block}

【本轮问题】
{query}
"""


def _fmt(memories: list[dict]) -> str:
    """将 recall 返回的记忆列表格式化为多行文本，供注入提示词。

    Args:
        memories: recall() 返回的列表，每项格式为 {"memory": str, "score": float}。

    Returns:
        str: 格式化的记忆文本，或提示"首次咨询"的说明文字。
    """
    # 若列表为空，说明该患者尚无历史记忆（首次咨询）
    if not memories:
        return "（暂无历史记忆，为首次咨询）"

    # 遍历记忆列表，过滤掉空 memory 字段，生成"序号. 内容"格式的行列表
    lines = [f"{i}. {m['memory'].strip()}" for i, m in enumerate(memories, 1) if m.get("memory")]

    # 拼接为多行字符串；若所有项均无 memory 字段则返回默认文字
    return "\n".join(lines) or "（暂无历史记忆）"


class MedicalAgent:
    """医疗健康咨询智能体。

    核心特性：
        - 优先调用工单12（Neo4j 知识图谱增强的医疗 Agent）获取高质量回复。
        - 工单12不可用时自动降级（fallback）到本地 LLM，保证服务可用性。
        - 每次对话前召回患者历史记忆，注入到查询中实现个性化诊断建议。
    """

    domain = "medical"  # 领域标识，用于构造 user_id 前缀（patient_xxx）

    def __init__(self) -> None:
        """初始化 LLM 客户端（fallback 用）和 MemoryClient 单例。"""
        self.llm    = LLMClient()              # fallback 时使用的本地 LLM 客户端
        self.memory = MemoryClient.instance()  # 全局记忆客户端单例（懒加载初始化）

    def chat(self, patient_id: str, query: str, *, session_id: Optional[str] = None) -> dict:
        """处理一次患者对话请求。

        Args:
            patient_id: 原始患者 ID（如 "p001"），不含前缀。
            query:      患者本次输入的问题文本。
            session_id: 可选会话 ID，写入 mem0 的 metadata 便于后续过滤。

        Returns:
            dict: {
                "reply":      str,        # 最终回复文本（来自工单12或本地 LLM）
                "recalled":   list[dict], # 召回的历史记忆条目
                "elapsed_ms": int,        # recall 检索耗时（毫秒）
                "source":     str,        # "wt12_neo4j" | "fallback_llm"
            }
        """
        # 构造带领域前缀的完整 user_id，如 "patient_p001"
        user_id = make_user_id(self.domain, patient_id)

        # 1. 召回历史记忆 ——————————————————————————————————————————
        t0 = time.perf_counter()  # 开始计时
        recalled = self.memory.recall(user_id, query, limit=5)  # 语义检索最多 5 条相关记忆
        elapsed_ms = int((time.perf_counter() - t0) * 1000)     # 计算耗时（毫秒）
        logger.info("[medical] patient=%s 召回 %d 条 %dms", patient_id, len(recalled), elapsed_ms)

        # 2. 把记忆注入到 query 里 ——————————————————————————————————
        memory_block = _fmt(recalled)  # 将记忆列表格式化为可读字符串
        # 拼接成完整的用户侧提示词：记忆块 + 本轮问题，发给工单12或本地 LLM
        full_query   = USER_TEMPLATE.format(memory_block=memory_block, query=query)

        # 3. 优先调工单12，失败则 fallback 到本地 LLM ——————————————
        reply, source = self._call_wt12(full_query, patient_id)  # 尝试调用工单12

        # 若工单12调用失败（reply 为 None），切换到 fallback 模式
        if reply is None:
            logger.warning("[medical] 工单12不可用，使用本地LLM fallback")
            # 使用本地 LLM（硅基流动 Qwen）生成回复
            reply  = self.llm.chat(system=FALLBACK_SYSTEM, user=full_query)
            source = "fallback_llm"  # 标记回复来源为本地 LLM fallback

        # 4. 异步写入本轮对话记忆（fire-and-forget，不阻塞当前请求） ——
        self.memory.remember(
            user_id=user_id,
            messages=[
                {"role": "user",      "content": query},   # 患者原始问题（不含注入的记忆块）
                {"role": "assistant", "content": reply},   # 最终回复文本
            ],
            # metadata 附加领域和会话信息，存入向量数据库
            metadata={"domain": self.domain, "session_id": session_id or "web"},
        )

        # 返回统一格式的结果字典
        return {"reply": reply, "recalled": recalled, "elapsed_ms": elapsed_ms, "source": source}

    def _call_wt12(self, full_query: str, patient_id: str) -> tuple[str | None, str]:
        """尝试调用工单12的 /chat 接口。

        向 WT12_URL 发送 POST 请求，携带注入了历史记忆的完整查询文本和患者 ID。
        工单12内部会执行意图识别、Neo4j 知识图谱查询，并返回增强后的回复。

        Args:
            full_query:  已注入历史记忆的完整提示文本。
            patient_id:  原始患者 ID（工单12用于查询患者相关图谱数据）。

        Returns:
            tuple: (reply, source)
                - 成功：(回复字符串, "wt12_neo4j")
                - 失败：(None, "")  调用者检测到 None 后触发 fallback
        """
        try:
            # httpx.post：发起同步 POST 请求，json 参数自动序列化为 JSON 并设置 Content-Type
            resp = httpx.post(
                WT12_URL,                                      # 工单12接口地址（来自环境变量）
                json={"query": full_query, "patient_id": patient_id},  # 请求体
                timeout=WT12_TIMEOUT,                         # 超时时间（秒）
            )
            # raise_for_status：若 HTTP 状态码为 4xx/5xx，抛出 HTTPStatusError 异常
            resp.raise_for_status()

            # 解析 JSON 响应体
            data  = resp.json()
            # 获取 reply 字段；若为空字符串也视为无效（用 or None 处理）
            reply = data.get("reply") or None

            # 若响应中没有 reply 字段（工单12返回格式异常），记录警告并触发 fallback
            if not reply:
                logger.warning("[medical] 工单12响应缺少 reply 字段，触发 fallback. resp=%s", data)
                return None, ""

            # 打印工单12额外返回的诊断信息（意图、实体、图谱数据条数）
            logger.info("[medical] wt12 intent=%s entity=%s graph=%d",
                        data.get("intent", ""),          # 意图识别结果（如 "symptom_query"）
                        data.get("entity", ""),          # 识别到的医学实体（如 "高血压"）
                        len(data.get("graph_data", []))) # Neo4j 图谱查询返回的节点/关系数量

            # 调用成功，返回回复内容和来源标识
            return reply, "wt12_neo4j"

        except httpx.ConnectError:
            # 连接错误：工单12服务未启动或端口不可达
            logger.warning("[medical] 工单12连接失败（未启动？）url=%s", WT12_URL)
        except httpx.TimeoutException:
            # 超时错误：工单12处理时间超过 WT12_TIMEOUT 秒
            logger.warning("[medical] 工单12超时 timeout=%ss", WT12_TIMEOUT)
        except Exception as e:
            # 其他异常（JSON 解析失败、HTTP 4xx/5xx 等）
            logger.warning("[medical] 工单12调用异常: %s", e)

        # 所有异常路径统一返回 (None, "")，由调用方触发 fallback
        return None, ""
