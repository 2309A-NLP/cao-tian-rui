"""
容错测试套件：验证无效输入不导致程序崩溃，并返回结构化错误信息。

覆盖场景：
  - TestValidation：纯本地校验（不打网络），测试 validate_query 函数
    · 空 query / 纯空白 / None
    · 超长输入（边界值：500字 vs 501字）
    · 纯符号 / 纯数字
  - TestAgentErrors：Agent 层容错（走完整流程但不崩溃）
    · 空/空白/符号/超长 → intent == "error"
    · 不存在的疾病 → LLM 兜底，不报错（需网络，可跳过）
  - TestRuleClassifier：规则分类器单元测试（不打网络，纯本地）
    · 各类意图关键词命中
    · 无实体时返回 None
    · 歧义问句返回 None（走 LLM）
    · 超长输入返回 None
"""
import pytest   # pytest 包：Python 最流行的测试框架，提供 assert 增强、skip、fixture 等
import sys      # 标准库：操作 Python 路径
import os       # 标准库：读取环境变量

# 将项目根目录插入 sys.path，确保 `from src.xxx` 可以正确导入
# __file__ 是本测试文件路径，dirname 两次得到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 导入被测对象
from src.agent import run_agent, validate_query          # Agent 主入口和校验函数
from src.intent_rules import classify as rule_classify   # 规则分类器


class TestValidation:
    """
    输入校验单元测试（不发起网络请求，纯本地逻辑验证）。
    测试 validate_query() 函数的边界行为。
    """

    def test_empty_query(self):
        """空字符串、纯空白、None 均应返回错误信息（非 None）。"""
        assert validate_query("") is not None       # 空字符串：应报错
        assert validate_query("   ") is not None    # 纯空白：应报错
        assert validate_query(None) is not None     # None 类型：应报错

    def test_too_long(self):
        """边界值测试：501字应报错，500字应通过。"""
        assert validate_query("啊" * 501) is not None  # 超出500字上限，应报错
        assert validate_query("啊" * 500) is None      # 恰好500字，应通过（边界值）

    def test_pure_symbols(self):
        """纯符号和纯数字（无中文/字母）应报错。"""
        assert validate_query("!!!???") is not None     # 纯符号
        assert validate_query("123456") is not None     # 纯数字（无医学意义）
        assert validate_query("...  ...") is not None   # 混合标点和空白

    def test_valid(self):
        """合法的医学问题和简短英文均应通过校验（返回 None 表示无错误）。"""
        assert validate_query("百日咳是什么？") is None  # 标准中文问句
        assert validate_query("hi") is None              # 短英文（含字母即通过）


class TestAgentErrors:
    """
    Agent 层容错测试（走完整调用流程，验证不崩溃且返回正确的错误结构）。
    这些测试不依赖网络（校验在进入 LLM 之前就已拦截）。
    """

    def test_empty_query_returns_error(self):
        """空字符串：应返回 intent=='error'，reply 包含提示词。"""
        r = run_agent("")
        assert r["intent"] == "error"                          # 意图必须是 error
        assert "空" in r["reply"] or "请" in r["reply"]        # 回复要给出提示

    def test_whitespace_query(self):
        """纯空白（含 tab/换行）：应被识别为空输入，返回 error。"""
        r = run_agent("   \t\n   ")
        assert r["intent"] == "error"

    def test_symbol_only_query(self):
        """纯符号输入：无法识别为有效医学问题，应返回 error。"""
        r = run_agent("!!!???")
        assert r["intent"] == "error"

    def test_super_long_query(self):
        """超出 500 字上限：应被拦截，返回 error（不调用 LLM）。"""
        r = run_agent("啊" * 600)
        assert r["intent"] == "error"

    @pytest.mark.skipif(os.getenv("SKIP_NETWORK") == "1", reason="需要 API")
    def test_nonexistent_disease(self):
        """
        图谱中不存在的疾病（如"肛门癌"）：
        - 不崩溃
        - intent 不为 error（能正常识别意图）
        - reply 不为空（LLM 用通用医学知识兜底）
        需要网络连接，设置环境变量 SKIP_NETWORK=1 可跳过。
        """
        r = run_agent("肛门癌是怎么治疗的？")
        # 兜底答案不为空、不崩溃
        assert r["reply"]              # 回复不为空
        assert r["intent"] != "error"  # 意图识别正常（即使图谱无数据）


class TestRuleClassifier:
    """
    规则分类器单元测试（不打网络，纯本地逻辑）。
    验证各类意图的关键词命中、实体提取，以及边界/歧义场景的正确处理。
    """

    def test_hit_complication(self):
        """含"并发症"关键词 → 命中 disease_to_complication，实体含"百日咳"。"""
        r = rule_classify("百日咳的并发症有哪些？")
        assert r is not None                          # 应命中（非 None）
        assert r[0] == "disease_to_complication"     # 意图正确
        assert "百日咳" in r[1]                       # 实体包含疾病名

    def test_hit_transmission(self):
        """含"怎么传播"关键词 → 命中 disease_to_transmission。"""
        r = rule_classify("乙肝怎么传播？")
        assert r is not None
        assert r[0] == "disease_to_transmission"

    def test_hit_drug(self):
        """含"用什么药"关键词 → 命中 disease_to_drug。"""
        r = rule_classify("流感用什么药？")
        assert r is not None
        assert r[0] == "disease_to_drug"

    def test_hit_diet_noeat(self):
        """含"不能吃什么"关键词 → 命中 disease_to_diet。"""
        r = rule_classify("糖尿病不能吃什么？")
        assert r is not None
        assert r[0] == "disease_to_diet"

    def test_hit_symptom_to_disease(self):
        """含"可能是什么病"关键词 → 命中 symptom_to_disease（反向推断）。"""
        r = rule_classify("头痛可能是什么病？")
        assert r is not None
        assert r[0] == "symptom_to_disease"

    def test_hit_dept(self):
        """含"挂什么科"关键词 → 命中 disease_to_dept。"""
        r = rule_classify("高血压挂什么科？")
        assert r is not None
        assert r[0] == "disease_to_dept"

    def test_hit_info_bloodtest(self):
        """含"血常规"关键词 → 命中 disease_info（综合信息/检查类）。"""
        r = rule_classify("百日咳患者的血常规有什么特征？")
        assert r is not None
        assert r[0] == "disease_info"

    def test_no_entity_returns_none(self):
        """问句无法提取有效实体时应返回 None（交给 LLM）。"""
        assert rule_classify("怎么办？") is None   # 无实体，实体提取失败
        assert rule_classify("嗯嗯") is None       # 非问句，无关键词也无实体

    def test_ambiguous_returns_none(self):
        """
        问句同时触发多个意图（用药 + 饮食）→ 规则放弃，返回 None 交给 LLM。
        这是"宁缺毋滥"原则：歧义时不强行分类。
        """
        r = rule_classify("糖尿病用什么药能吃什么？")
        assert r is None  # 歧义：同时命中 drug 和 diet

    def test_too_long_returns_none(self):
        """超过 200 字的问句规则分类器直接放弃（避免慢正则）。"""
        assert rule_classify("啊" * 300) is None
