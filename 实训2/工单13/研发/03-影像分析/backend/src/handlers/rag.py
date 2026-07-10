"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
RAG Handler：影像+文本 → 直接用 query 检索医学知识 → 增强回答。

RAG（Retrieval Augmented Generation，检索增强生成）是一种将
知识库检索与大语言模型生成结合的技术，可以减少模型"幻觉"并引入外部知识。

本处理器的简化 RAG 流程（去掉了 Caption 步骤）：
1. 直接用原始 query 向量检索 ChromaDB，取 top-K 相关文档
2. 若命中文档，将参考片段拼入 System Prompt，让 VLM 结合影像作答；
   若无命中，退化为纯 VQA（直接基于影像作答，并说明依据）
"""

# Optional：类型提示，表示可以为 None
from typing import Optional

# 从父包导入所需模块（使用相对导入）
from ..config import DISCLAIMER      # 医疗免责声明文本
from ..logger import get_logger      # 日志工厂函数
from ..models import RefDoc          # 参考文档数据模型
from ..rag_store import query as rag_query  # 向量库检索函数（重命名避免与参数冲突）
from ..vlm_client import get_vlm_client    # VLM 客户端工厂函数

# 获取本模块专用的日志记录器
logger = get_logger("wt13.rag")

# ── RAG 任务的系统提示词 ──
# 告知 VLM 如何结合参考文档和影像作答
RAG_ANSWER_SYSTEM = (
    "你是一位经验丰富的医学影像分析助手。请结合参考知识和影像回答用户的问题。\n"
    "回答规则：\n"
    "1) 优先基于影像可观察内容，参考知识仅作为背景补充；\n"
    "2) 若参考知识与影像所见冲突，以影像为准，可指出参考知识的适用范围；\n"
    "3) 若参考知识与问题相关，请在关键结论后用 [文档N] 形式标注引用；\n"
    "4) 中文回答，用词准确、简洁；\n"
    "5) 涉及诊断或建议时，附上：本回答仅供参考，不构成医疗诊断建议。"
)


class RAGHandler:
    """
    检索增强生成处理器。

    结合向量知识库检索和 VLM 图文理解，提供有知识背书的医疗影像问答。
    """

    def __init__(self):
        """初始化处理器，获取 VLM 客户端单例。"""
        self.vlm = get_vlm_client()  # 获取全局 VLM 客户端（单例）

    def run(
        self,
        image_b64: str,
        query: str,
        image_format: str = "jpeg",
        top_k: Optional[int] = None,
    ) -> tuple[str, list[RefDoc]]:
        """
        执行 RAG 任务：检索知识库 + 结合影像生成增强回答。

        参数：
            image_b64 (str)：图片的 base64 编码字符串
            query (str)：用户的问题文本
            image_format (str)：图片格式（"jpeg" 或 "png"），默认 "jpeg"
            top_k (Optional[int])：覆盖默认的检索文档数量，None 表示使用配置默认值

        返回值：
            tuple[str, list[RefDoc]]：
            - str：模型生成的回答文本（已追加免责声明）
            - list[RefDoc]：从知识库中检索到的参考文档列表（可能为空）
        """
        # ── 步骤1：记录检索日志 ──
        logger.info(
            "RAG 检索文本",
            extra={"payload": {"query": query[:120]}},  # 只记录前 120 个字符，避免日志过大
        )

        # ── 步骤2：向量相似度检索 ──
        # 直接用原始用户问题作为检索文本（简化流程，省去影像描述步骤）
        # top_k 若为 None 则使用默认值 5（来自 config.TOP_K）
        refs = rag_query(query, top_k=top_k if top_k is not None else 5)

        # ── 步骤3：组装增强上下文 Prompt ──
        if refs:
            # 有命中文档：构建包含参考内容的 Prompt
            ref_lines = []
            for i, r in enumerate(refs, 1):
                # 将每条文档格式化为 [文档N] (相似度) 内容 的格式
                ref_lines.append(f"[文档{i}] (相似度 {r.score})\n{r.snippet}")

            # 将所有参考文档合并为一个上下文字符串（双换行分隔各文档）
            context = "\n\n".join(ref_lines)

            # 构建完整用户 Prompt：问题 + 检索到的知识 + 回答指令
            user_prompt = (
                f"用户问题：{query}\n\n"
                f"以下是从医学知识库中检索到的相关参考：\n\n{context}\n\n"
                f"请结合影像与参考回答用户问题，并在需要的关键结论后用 [文档N] 标注引用。"
            )
        else:
            # 无命中文档：退化为纯影像问答，明确告知用户此回答没有知识库支撑
            user_prompt = (
                f"用户问题：{query}\n\n"
                "（知识库无相关命中，直接基于影像作答，并说明这是仅基于影像的判断。）"
            )

        # ── 步骤4：调用 VLM 生成最终回答 ──
        answer = self.vlm.chat_vision(
            image_b64=image_b64,
            prompt=user_prompt,
            system=RAG_ANSWER_SYSTEM,  # 使用 RAG 专用系统提示词
            max_tokens=1200,           # 给予充足的 token 预算（RAG 回答可能较长）
            temperature=0.3,           # 中等温度：保持一定创造性但不过于随机
            image_format=image_format,
        )

        # ── 步骤5：确保免责声明存在 ──
        # 若模型回答中未包含免责声明，则自动追加
        # "仅供参考" 和 "遵医嘱" 是系统提示词中要求模型自行添加的关键词
        if "仅供参考" not in answer and "遵医嘱" not in answer:
            answer = f"{answer}\n\n⚠️ {DISCLAIMER}"

        # 返回回答文本和检索到的参考文档列表
        return answer, refs
