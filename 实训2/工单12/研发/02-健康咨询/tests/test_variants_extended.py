"""
20+ 扩展变体测试（覆盖工单要求 30+ 变体场景）。

覆盖范围：
  - 更多常见疾病（心脑血管/消化/儿科/呼吸/皮肤/五官）
  - 症状反查（多症状组合）
  - 组合意图问句

每个测试用例：发送问题 → 检查回复是否包含预期关键词之一。
API 限流时自动 skip（不算失败）。

运行方法：
    .venv\\Scripts\\python -m pytest tests/test_variants_extended.py -v
"""
import pytest   # pytest 包：测试框架，提供 skip / assert 增强等功能
import time     # 标准库：控制测试间隔（防 API 限流）
import sys      # 标准库：操作 Python 导入路径
import os       # 标准库：读取系统路径

# 将项目根目录加入 sys.path，使 `from src.xxx` 导入生效
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_agent  # 被测 Agent 主入口

_DELAY = 1.0  # 每个测试用例之间的间隔秒数（防硅基流动 API 限流）


def _check(query: str, keywords: list) -> tuple:
    """
    执行一次 Agent 查询并检查关键词命中。

    参数:
        query (str): 用户问题
        keywords (list): 预期关键词列表（回复包含其中任一即 PASS）

    返回:
        tuple(matched: bool, result: dict)
          matched: 是否命中任一关键词
          result:  run_agent() 完整返回值（用于 assert 打印详情）
    """
    time.sleep(_DELAY)               # 限流间隔
    result = run_agent(query)        # 发起 Agent 查询
    reply = result["reply"].lower()  # 转小写便于不区分大小写匹配

    # API 限流时 reply 会包含特定提示词，自动 skip 避免误报失败
    if "暂时不可用" in reply or "请稍后再试" in reply:
        pytest.skip("API 限流，跳过本次用例")

    # 检查回复中是否包含任一预期关键词（也转小写比较）
    matched = any(kw.lower() in reply for kw in keywords)
    return matched, result


class TestExtendedDiseases:
    """
    20 个不同疾病/意图的扩展变体测试。
    覆盖心脑血管、消化、儿科、呼吸、皮肤、五官等多学科常见疾病。
    """

    def test_e01_stroke_symptom(self):
        """脑卒中症状查询（病名含"脑"，常见关键词：偏瘫/口角歪斜/言语障碍）。"""
        ok, r = _check("脑卒中有哪些症状？",
                        ["偏瘫", "口角歪斜", "言语", "意识", "头痛", "肢体"])
        assert ok, f"回复：{r['reply']}"

    def test_e02_gastritis_diet(self):
        """胃炎饮食建议（应提到清淡/易消化食物）。"""
        ok, r = _check("胃炎能吃什么食物？",
                        ["清淡", "易消化", "粥", "面食", "蔬菜"])
        assert ok, f"回复：{r['reply']}"

    def test_e03_pneumonia_dept(self):
        """肺炎就诊科室（应提到呼吸科/内科/感染科）。"""
        ok, r = _check("肺炎应该挂什么科？",
                        ["呼吸", "内科", "感染"])
        assert ok, f"回复：{r['reply']}"

    def test_e04_measles_transmission(self):
        """麻疹传播途径（应提到飞沫/空气传播/呼吸道）。"""
        ok, r = _check("麻疹是怎么传染的？",
                        ["飞沫", "空气", "呼吸道", "接触"])
        assert ok, f"回复：{r['reply']}"

    def test_e05_hepatitis_prevent(self):
        """乙肝预防措施（应提到疫苗接种/母婴阻断）。"""
        ok, r = _check("乙肝如何预防？",
                        ["疫苗", "接种", "母婴阻断", "防"])
        assert ok, f"回复：{r['reply']}"

    def test_e06_diabetes_complication(self):
        """糖尿病并发症（应提到肾病/视网膜/神经病变/足病等）。"""
        ok, r = _check("糖尿病有哪些并发症？",
                        ["肾病", "视网膜", "神经", "酮症", "心血管", "足"])
        assert ok, f"回复：{r['reply']}"

    def test_e07_child_fever_nursing(self):
        """小儿感冒发烧护理（应提到物理降温/退烧/多喝水）。"""
        ok, r = _check("小儿感冒发烧怎么护理？",
                        ["物理降温", "退烧", "多喝水", "休息", "温水"])
        assert ok, f"回复：{r['reply']}"

    def test_e08_asthma_drug(self):
        """哮喘用药（应提到沙丁胺醇/糖皮质激素/吸入剂等支气管扩张药）。"""
        ok, r = _check("哮喘患者常用什么药？",
                        ["沙丁胺醇", "支气管扩张", "糖皮质激素", "布地奈德", "吸入剂"])
        assert ok, f"回复：{r['reply']}"

    def test_e09_hypertension_diet(self):
        """高血压饮食禁忌（应提到盐/高盐/腌制/高脂等）。"""
        ok, r = _check("高血压不能吃什么？",
                        ["盐", "高盐", "腌制", "高脂", "钠", "油腻"])
        assert ok, f"回复：{r['reply']}"

    def test_e10_symptom_headache(self):
        """症状反查：突然剧烈头痛可能是什么病（应提到脑出血/偏头痛/颅内等）。"""
        ok, r = _check("突然剧烈头痛可能是什么病？",
                        ["脑出血", "脑血管", "偏头痛", "蛛网膜", "颅内", "血压"])
        assert ok, f"回复：{r['reply']}"

    def test_e11_symptom_chest_pain(self):
        """症状反查：胸痛可能由什么病引起（应提到心脏/冠心/肺等）。"""
        ok, r = _check("胸痛可能是什么病引起的？",
                        ["心", "冠心", "心肌", "心绞痛", "肺", "肋间"])
        assert ok, f"回复：{r['reply']}"

    def test_e12_symptom_diarrhea(self):
        """症状反查：腹泻可能是什么原因（应提到肠炎/感染/食物等）。"""
        ok, r = _check("腹泻可能是什么原因？",
                        ["肠", "感染", "食物", "痢疾", "肠炎", "细菌", "病毒"])
        assert ok, f"回复：{r['reply']}"

    def test_e13_covid_symptom(self):
        """新冠症状查询（应提到发热/咳嗽/乏力/呼吸困难/肺炎）。"""
        ok, r = _check("新型冠状病毒肺炎有什么症状？",
                        ["发热", "咳嗽", "乏力", "呼吸", "肺炎"])
        assert ok, f"回复：{r['reply']}"

    def test_e14_kidney_stone_treat(self):
        """肾结石治疗方案（应提到排石/碎石/手术/多喝水等）。"""
        ok, r = _check("肾结石怎么治疗？",
                        ["排石", "碎石", "手术", "药物", "多喝水", "体外冲击波"])
        assert ok, f"回复：{r['reply']}"

    def test_e15_gerd_cause(self):
        """胃食管反流病因（应提到食管下段括约肌/胃酸/反流等）。"""
        ok, r = _check("胃食管反流病是什么原因引起的？",
                        ["食管", "下段", "括约肌", "压力", "反流", "胃酸"])
        assert ok, f"回复：{r['reply']}"

    def test_e16_conjunctivitis_dept(self):
        """结膜炎就诊科室（应提到眼科/五官科）。"""
        ok, r = _check("结膜炎应该看哪个科？",
                        ["眼科", "五官"])
        assert ok, f"回复：{r['reply']}"

    def test_e17_urticaria_treat(self):
        """荨麻疹治疗（应提到抗组胺药/氯雷他定/西替利嗪等抗过敏药）。"""
        ok, r = _check("荨麻疹怎么治？",
                        ["抗组胺", "氯雷他定", "西替利嗪", "激素", "抗过敏"])
        assert ok, f"回复：{r['reply']}"

    def test_e18_migraine_prevent(self):
        """偏头痛预防（应提到规律作息/避免诱因/情绪管理等）。"""
        ok, r = _check("偏头痛怎么预防发作？",
                        ["休息", "睡眠", "诱因", "避免", "咖啡", "情绪", "规律"])
        assert ok, f"回复：{r['reply']}"

    def test_e19_gallstone_symptom(self):
        """胆结石症状（应提到右上腹痛/绞痛/黄疸等）。"""
        ok, r = _check("胆结石有哪些症状？",
                        ["右上腹", "腹痛", "绞痛", "黄疸", "恶心"])
        assert ok, f"回复：{r['reply']}"

    def test_e20_pericarditis_intro(self):
        """心包炎疾病介绍（应提到心包/炎症/心脏/积液等）。"""
        ok, r = _check("心包炎是什么病？",
                        ["心包", "炎症", "心脏", "积液", "感染"])
        assert ok, f"回复：{r['reply']}"
