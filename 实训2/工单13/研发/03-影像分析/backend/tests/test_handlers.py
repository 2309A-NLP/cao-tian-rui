"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
Handler 单元测试（无需运行服务，mock VLM + RAG store）

本文件测试三个 Handler 的内部逻辑，完全不依赖外部服务：
- TestMRGHandler：测试 MRGHandler 的 JSON 解析、重试、降级机制
- TestRAGHandler：测试 RAGHandler 的检索命中/未命中两条路径及免责声明
- TestTryParseJson：单独测试 mrg._try_parse_json 的三种解析策略

运行：cd 13--工单13 && python -m pytest tests/test_handlers.py -v
"""

# base64：Python 内置模块，用于解码测试用的最小像素图片
import base64

# json：Python 内置模块，用于构造 VLM mock 返回的 JSON 字符串
import json

# sys：Python 内置模块，用于修改模块搜索路径
import sys

# os：Python 内置模块（此处导入但未使用）
import os

# unittest.mock：Python 内置 mock 测试工具
# MagicMock：自动创建的 mock 对象，支持任意属性访问和方法调用记录
# patch：临时替换模块中的对象为 mock
from unittest.mock import MagicMock, patch

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# pytest：Python 测试框架
import pytest

# 将 backend/ 加入模块搜索路径，确保 "from src.xxx import" 能正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── 最小像素 PNG (1×1 白色) 的 base64 字符串 ─────────────────────
# 此字符串是一个 1×1 白色像素 PNG 图片的 base64 编码
# 用于替代真实医疗影像作为 image_b64 参数传入 Handler（Handler 内部不会实际解析图片）
_1PX_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


# ════════════════════════════════════════════════
# MRG Handler 单元测试
# ════════════════════════════════════════════════

class TestMRGHandler:
    """
    测试 MRGHandler.run() 的三种执行路径：
    1. 正常路径：VLM 直接返回合法 JSON
    2. Markdown 包装：JSON 被 ```json...``` 包裹
    3. 熔断降级：连续 3 次解析失败
    4. 重试成功：第1次失败，第2次成功
    5. extra_context 传参验证
    """

    def _make_handler(self, vlm_responses: list[str]):
        """
        构造一个使用 mock VLM 的 MRGHandler 实例。

        使用 __new__ 绕过 __init__（避免触发 get_vlm_client() 初始化），
        然后手动设置 vlm 属性为 mock 对象。

        参数：
            vlm_responses (list[str])：chat_vision 依次返回的字符串列表
                                        （side_effect 会按顺序消费列表中的元素）

        返回值：
            MRGHandler：带 mock VLM 的处理器实例
        """
        from src.handlers.mrg import MRGHandler
        # __new__ 创建实例但不调用 __init__（避免初始化副作用）
        handler = MRGHandler.__new__(MRGHandler)
        mock_vlm = MagicMock()
        # side_effect 设为列表：每次调用依次返回列表中的下一个元素
        mock_vlm.chat_vision.side_effect = vlm_responses
        handler.vlm = mock_vlm
        return handler

    # ── 测试1：VLM 直接返回合法 JSON ──────────────────────────────

    def test_mrg_direct_json(self):
        """VLM 返回裸 JSON（无包装）→ report 四字段全部正确填充，文本包含标题。"""
        # 构造合法的报告 JSON 字符串
        raw = json.dumps({
            "chief_complaint": "胸痛 2 天",
            "findings": "双肺纹理清晰，未见明显渗出影。",
            "impression": "未见急性病变",
            "recommendation": "结合临床，必要时复查。",
        }, ensure_ascii=False)  # ensure_ascii=False：保留中文字符
        handler = self._make_handler([raw])  # VLM 只被调用一次
        report, text = handler.run(_1PX_PNG_B64)

        # 验证 report 对象的字段值
        assert report.chief_complaint == "胸痛 2 天"
        assert "双肺" in report.findings      # 使用 in 检查子串存在
        assert report.impression != ""        # impression 非空
        assert report.recommendation != ""   # recommendation 非空
        assert "影像所见" in text             # 格式化文本中包含节标题

    # ── 测试2：VLM 返回 Markdown 代码块包裹的 JSON ────────────────

    def test_mrg_markdown_json(self):
        """VLM 用 ```json...``` 包裹 JSON → _try_parse_json 的策略2 应能正确提取并解析。"""
        inner = json.dumps({
            "chief_complaint": "头痛",
            "findings": "未见明显异常信号。",
            "impression": "基本正常",
            "recommendation": "随访观察。",
        }, ensure_ascii=False)
        # 模拟 VLM 输出了 Markdown 代码块格式
        raw = f"```json\n{inner}\n```"
        handler = self._make_handler([raw])
        report, text = handler.run(_1PX_PNG_B64)

        assert report.chief_complaint == "头痛"
        assert "基本正常" in report.impression

    # ── 测试3：VLM 连续 3 次返回非 JSON → 熔断降级 ──────────────

    def test_mrg_fallback_degradation(self):
        """
        3 次 JSON 解析全部失败 → 触发熔断降级：
        - findings 填充 VLM 最后一次的原始文本
        - impression 包含"解析失败"或"降级"关键词
        """
        bad_response = "非常抱歉，我无法生成报告格式。"
        # side_effect 列表长度为 3，与 _MAX_PARSE_RETRIES=3 对应
        handler = self._make_handler([bad_response] * 3)
        report, text = handler.run(_1PX_PNG_B64)

        # 降级时：原始输出应在 findings 中
        assert bad_response in report.findings
        # 降级标识：impression 应包含降级说明
        assert "解析失败" in report.impression or "降级" in report.impression

    # ── 测试4：第1次失败第2次成功 → retry 有效 ───────────────────

    def test_mrg_retry_success_on_second(self):
        """第1次返回乱码，第2次返回合法 JSON → 最终应成功解析（不触发降级）。"""
        bad = "这是一段无效内容..."  # 第1次调用返回这个（会触发重试）
        good = json.dumps({
            "chief_complaint": "",
            "findings": "重试后成功的所见。",
            "impression": "重试成功",
            "recommendation": "",
        }, ensure_ascii=False)         # 第2次调用返回这个（JSON 合法）
        handler = self._make_handler([bad, good])
        report, text = handler.run(_1PX_PNG_B64)

        # 最终应使用第2次的合法 JSON 结果
        assert "重试成功" in report.impression

    # ── 测试5：extra_context 会拼进 prompt ────────────────────────

    def test_mrg_extra_context_passed(self):
        """传入 extra_context → chat_vision 的 prompt 参数中应包含该信息。"""
        raw = json.dumps({
            "chief_complaint": "测试",
            "findings": "正常",
            "impression": "正常",
            "recommendation": "随访",
        }, ensure_ascii=False)
        handler = self._make_handler([raw])
        handler.run(_1PX_PNG_B64, extra_context="68岁男性，肺结节随访")

        # 获取 chat_vision 被调用时的参数
        call_kwargs = handler.vlm.chat_vision.call_args
        # 兼容位置参数和关键字参数两种调用方式
        prompt = call_kwargs.kwargs.get("prompt", "") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else ""
        )
        # extra_context 应被拼接到 prompt 中
        assert "68岁" in prompt or "肺结节" in prompt


# ════════════════════════════════════════════════
# RAG Handler 单元测试
# ════════════════════════════════════════════════

class TestRAGHandler:
    """
    测试 RAGHandler.run() 的两条主路径：
    1. 有检索命中：回答中应含 [文档N] 引用标注
    2. 无检索命中：仍能回答（退化为纯影像分析）
    3. 免责声明自动追加
    4. caption 失败的容错路径（若 RAGHandler 有 caption 步骤的话）
    """

    def _make_handler(self, caption: str, refs: list, answer: str):
        """
        构造带 mock VLM 和 mock rag_query 的 RAGHandler 实例。

        当前实现中 RAGHandler.run() 只调用一次 chat_vision（直接用 query 检索，无 caption 步骤）。
        caption 参数保留是为了兼容旧有测试结构，实际上 side_effect 只消费第二个元素（answer）。

        参数：
            caption (str)：历史遗留参数（第1次 chat_vision 调用的返回值，当前实现可能不调用）
            refs (list[dict])：rag_query mock 返回的文档配置列表，格式：
                               [{"snippet": "...", "score": 0.9}, ...]
            answer (str)：chat_vision 最终返回的回答文本

        返回值：
            tuple[RAGHandler, list[RefDoc]]：(处理器实例, RefDoc 对象列表)
        """
        from src.handlers.rag import RAGHandler
        from src.models import RefDoc

        # 绕过 __init__ 创建实例
        handler = RAGHandler.__new__(RAGHandler)
        mock_vlm = MagicMock()
        # side_effect 包含两个值：先 caption，后 answer（当前实现只用 answer）
        mock_vlm.chat_vision.side_effect = [caption, answer]
        handler.vlm = mock_vlm

        # 将 dict 列表转换为 RefDoc 对象列表（与真实 rag_query 返回格式一致）
        ref_objects = [
            RefDoc(doc_id=f"doc{i}", title=r.get("title", ""), snippet=r["snippet"], score=r["score"])
            for i, r in enumerate(refs, 1)
        ]
        return handler, ref_objects

    def _run_with_mock_rag(self, handler, refs, image_b64, query):
        """
        patch rag_query 后调用 handler.run()，避免实际连接 ChromaDB。

        参数：
            handler：RAGHandler 实例
            refs：mock rag_query 的返回值（RefDoc 列表）
            image_b64 (str)：测试用图片 base64 字符串
            query (str)：测试查询文本

        返回值：
            tuple[str, list[RefDoc]]：handler.run() 的返回值
        """
        # patch "src.handlers.rag.rag_query"：替换 rag.py 中导入的 rag_query 函数
        with patch("src.handlers.rag.rag_query", return_value=refs):
            return handler.run(image_b64, query)

    # ── 测试1：有命中参考 → 回答中应有 [文档N] 标注 ────────────────────

    def test_rag_with_refs_answer_has_citation(self):
        """知识库有命中 → 模型回答中应包含 [文档1] 或 [文档2] 引用标注。"""
        refs_data = [
            {"snippet": "胸部 X 线可见肺纹理增粗。", "score": 0.91},
            {"snippet": "CT 扫描对肺结节敏感性更高。", "score": 0.85},
        ]
        # 模拟 VLM 返回了包含文档引用标注的回答
        answer_text = "根据影像所见，考虑肺纹理增粗 [文档1]，建议进一步 CT 检查 [文档2]。"
        handler, ref_objects = self._make_handler("胸部X线 纹理增粗", refs_data, answer_text)
        answer, refs = self._run_with_mock_rag(handler, ref_objects, _1PX_PNG_B64, "影像表现如何？")

        # 验证回答中含引用标注
        assert "[文档1]" in answer or "[文档2]" in answer
        # 验证参考文档列表长度正确
        assert len(refs) == 2

    # ── 测试2：无命中 → 仍能回答，refs 为空 ─────────────────

    def test_rag_no_refs_still_answers(self):
        """知识库无命中 → 不应报错，answer 非空，refs 为空列表。"""
        answer_text = "仅基于影像判断，未检索到相关知识库内容。建议临床进一步确认。"
        handler, _ = self._make_handler("腹部 CT 异常影像", [], answer_text)
        answer, refs = self._run_with_mock_rag(handler, [], _1PX_PNG_B64, "腹部有什么异常？")

        assert answer != ""     # 答案不能为空
        assert refs == []       # 无命中时应返回空列表

    # ── 测试3：caption 失败仍能用原 query 检索 ─────────────────────

    def test_rag_caption_failure_fallback(self):
        """
        第1次 chat_vision 调用抛异常（模拟超时/API错误）→
        RAGHandler 应退化为使用原始 query 检索，第2次 chat_vision 正常回答。

        注意：当前 RAGHandler 实现没有显式的 caption 步骤，
        此测试验证即使 chat_vision 首次失败，也能正常处理（取决于实现细节）。
        """
        from src.handlers.rag import RAGHandler
        from src.models import RefDoc

        handler = RAGHandler.__new__(RAGHandler)
        mock_vlm = MagicMock()
        # 第一次调用抛异常，第二次调用返回正常答案
        mock_vlm.chat_vision.side_effect = [
            Exception("VLM 超时"),
            "基于影像判断，未见明显异常。",
        ]
        handler.vlm = mock_vlm

        # 构造 mock 参考文档
        ref_objects = [
            RefDoc(doc_id="d1", title="", snippet="参考片段", score=0.7)
        ]
        with patch("src.handlers.rag.rag_query", return_value=ref_objects):
            answer, refs = handler.run(_1PX_PNG_B64, "影像正常吗？")

        assert "未见明显异常" in answer  # 第2次调用的答案
        assert len(refs) == 1            # 检索到 1 条参考

    # ── 测试4：免责声明自动追加 ─────────────────────────────────

    def test_rag_disclaimer_appended_when_missing(self):
        """
        当 VLM 回答不含'仅供参考'或'遵医嘱'时，
        RAGHandler 应自动在答案末尾追加免责声明（⚠️）。
        """
        answer_text = "影像显示肺野清晰。"  # 纯描述，无免责声明
        handler, _ = self._make_handler("", [], answer_text)
        answer, _ = self._run_with_mock_rag(handler, [], _1PX_PNG_B64, "肺部情况？")

        # 验证免责声明已被追加（三种形式之一存在即可）
        assert "⚠️" in answer or "仅供参考" in answer or "遵医嘱" in answer


# ════════════════════════════════════════════════
# MRG _try_parse_json 内部函数单元测试
# ════════════════════════════════════════════════

class TestTryParseJson:
    """
    测试 mrg._try_parse_json 的三种解析策略（白盒测试内部私有函数）。

    通过直接导入并调用内部函数，验证各解析路径的正确性。
    """

    def setup_method(self):
        """
        每个测试方法执行前，导入并缓存 _try_parse_json 函数。

        setup_method 是 pytest 的 fixture 机制，在每个测试方法前自动调用。
        """
        from src.handlers.mrg import _try_parse_json
        self.parse = _try_parse_json  # 将函数引用保存到实例属性

    def test_direct_json(self):
        """策略1：整个文本是合法 JSON → 直接解析成功。"""
        data = {"chief_complaint": "x", "findings": "y", "impression": "z", "recommendation": "r"}
        result = self.parse(json.dumps(data))  # json.dumps：将字典序列化为 JSON 字符串
        assert result == data  # 解析结果应与原始字典完全一致

    def test_markdown_code_block(self):
        """策略2：JSON 被 ```json...``` 包裹 → 通过正则提取并解析成功。"""
        data = {"findings": "test findings"}
        # 构造 Markdown 代码块格式
        raw = f"```json\n{json.dumps(data)}\n```"
        result = self.parse(raw)
        assert result == data  # 提取并解析结果应正确

    def test_json_embedded_in_text(self):
        """策略3：JSON 对象嵌入在普通文本中 → 通过查找 {} 提取并解析成功。"""
        data = {"impression": "正常"}
        # JSON 对象前后有其他文字
        raw = f"以下是报告：\n{json.dumps(data)}\n请参考。"
        result = self.parse(raw)
        assert result == data  # 应正确提取 {} 之间的内容

    def test_empty_returns_none(self):
        """空字符串输入 → 应返回 None（早期返回，不尝试任何解析）。"""
        assert self.parse("") is None

    def test_invalid_json_returns_none(self):
        """纯文本（非 JSON 格式）→ 三种策略全部失败，返回 None。"""
        assert self.parse("这不是json") is None
