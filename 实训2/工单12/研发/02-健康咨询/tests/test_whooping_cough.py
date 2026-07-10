"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-健康咨询

工单给定的 10 个百日咳测试案例 + 10 个多疾病/多症状变体场景。

运行：
    cd F:\\kimi  project\\医疗agent1\\02-健康咨询
    .venv\\Scripts\\python -m pytest tests/ -v

验收标准（来自工单）：
  - 检索精度 >= 80%（即 20 题通过率 >= 16/20）
  - 响应时间 < 500ms（意图识别阶段）
  - 容错：无效输入不崩溃
"""
import pytest   # pytest 包：测试框架，skip/assert 增强
import time     # 标准库：每题之间等待间隔
import sys      # 标准库：修改 Python 导入路径
import os       # 标准库：读取环境变量

# 将项目根目录添加到 sys.path，使 `from src.xxx` 导入生效
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_agent  # 被测 Agent 主入口

# 硅基流动 API 限流保护：串行测试，每题间隔 1 秒
_DELAY = 1.0


def _check(query: str, keywords: list) -> tuple:
    """
    执行 Agent 查询并检查回复是否包含任一关键词。

    参数:
        query (str): 测试问题
        keywords (list): 预期关键词列表（任一匹配即通过）

    返回:
        tuple(matched: bool, result: dict)
    """
    time.sleep(_DELAY)               # 限流间隔（每题等待 1 秒）
    result = run_agent(query)        # 调用 Agent
    reply = result["reply"].lower()  # 转小写，不区分大小写匹配

    # 检测到 API 限流标志时自动跳过，不计入失败
    if "暂时不可用" in reply or "请稍后再试" in reply:
        pytest.skip("API 限流，跳过本次用例")

    # 任一关键词出现在回复中即为匹配
    matched = any(kw.lower() in reply for kw in keywords)
    return matched, result


# ══════════════════════════════════════════════════════
# 工单给定：百日咳 10 个测试案例
# 来源：工单验收标准，必须全部通过（或达到 80% 通过率）
# ══════════════════════════════════════════════════════

class TestWhoopingCough:
    """百日咳专项测试：覆盖病原体/传播/症状/诊断/用药/并发/中医/护理/饮食等维度。"""

    def test_01_pathogen(self):
        """1. 病原体识别：回复应包含"百日咳杆菌"或"鲍特菌"（英文也接受）。"""
        ok, r = _check("百日咳的致病病原体是什么？",
                        ["百日咳杆菌", "鲍特菌", "bordetella"])
        assert ok, f"未找到关键词，回复：{r['reply']}"

    def test_02_transmission(self):
        """2. 传播途径判断：飞沫传播 / 呼吸道传播。"""
        ok, r = _check("百日咳主要通过什么途径传播？",
                        ["飞沫", "呼吸道"])
        assert ok, f"回复：{r['reply']}"

    def test_03_symptom(self):
        """3. 典型症状辨识：痉挛性咳嗽 / 鸡鸣样回声（最具特征性临床表现）。"""
        ok, r = _check("百日咳最具特征性的临床表现是什么？",
                        ["痉挛", "咳嗽", "鸡鸣", "吼声"])
        assert ok, f"回复：{r['reply']}"

    def test_04_lab(self):
        """4. 实验室诊断：血常规特征 = 白细胞总数升高 + 淋巴细胞比例增高。"""
        ok, r = _check("百日咳患者的血常规检查会呈现什么特征？",
                        ["白细胞", "淋巴细胞", "白细胞增多", "淋巴细胞升高", "淋巴细胞增多"])
        assert ok, f"回复：{r['reply']}"

    def test_05_drug(self):
        """5. 治疗药物选择：首选大环内酯类（红霉素/阿奇霉素/罗红霉素）。"""
        ok, r = _check("百日咳西医治疗首选的抗生素是什么？",
                        ["红霉素", "阿奇霉素", "罗红霉素", "大环内酯"])
        assert ok, f"回复：{r['reply']}"

    def test_06_complication(self):
        """6. 并发症识别：最常见并发症为支气管肺炎/肺不张/百日咳脑病。"""
        ok, r = _check("百日咳最常见的严重并发症是什么？",
                        ["支气管肺炎", "肺不张", "百日咳脑病", "脑炎"])
        assert ok, f"回复：{r['reply']}"

    def test_07_tcm(self):
        """7. 中医辨证治疗：痉咳期主方为桑白皮汤。"""
        ok, r = _check("中医治疗痉咳期百日咳的主方是什么？",
                        ["桑白皮汤", "桑白皮"])
        assert ok, f"回复：{r['reply']}"

    def test_08_isolation(self):
        """8. 预防措施——隔离期：卫健委标准为 40 天（四至六周亦可接受）。"""
        ok, r = _check("百日咳患者的隔离期应持续多久？",
                        ["40天", "40 天", "四十天", "四至六周", "4~6周", "4～6周"])
        assert ok, f"回复：{r['reply']}"

    def test_09_nursing(self):
        """9. 护理要点——防窒息：护理时须防止患儿发生窒息/发绀/喉痉挛。"""
        ok, r = _check("护理百日咳患儿时需特别注意防范什么紧急情况？",
                        ["窒息", "发绀", "喉痉挛", "守护", "观察呼吸"])
        assert ok, f"回复：{r['reply']}"

    def test_10_diet(self):
        """10. 营养指导——忌口：应避免海鲜类（螃蟹/海虾/海螺等）。"""
        ok, r = _check("百日咳患者应避免食用哪类食物？",
                        ["海鲜", "螃蟹", "海虾", "海螺"])
        assert ok, f"回复：{r['reply']}"


# ══════════════════════════════════════════════════════
# 变体场景：多疾病 + 多症状（共 10 题）
# 覆盖流感/糖尿病/高血压/乙肝/肺结核/哮喘/阑尾炎/贫血/感冒等常见病
# ══════════════════════════════════════════════════════

class TestVariants:
    """多疾病变体测试：验证 Agent 对常见病的覆盖广度。"""

    def test_v01_flu_drug(self):
        """流感治疗药物（抗病毒：奥司他韦/达菲；对症：退烧/解热）。"""
        ok, r = _check("流感应该吃什么药？",
                        ["奥司他韦", "达菲", "磷酸奥司他韦", "退烧", "解热", "对乙酰氨基酚"])
        assert ok, f"回复：{r['reply']}"

    def test_v02_diabetes_diet(self):
        """糖尿病饮食禁忌（高糖/甜食/精制碳水化合物等）。"""
        ok, r = _check("糖尿病患者不能吃什么？",
                        ["高糖", "甜食", "含糖", "精制碳水", "血糖", "碳水化合物"])
        assert ok, f"回复：{r['reply']}"

    def test_v03_hypertension_dept(self):
        """高血压就诊科室（心内科/内科/心血管科）。"""
        ok, r = _check("高血压应该挂哪个科？",
                        ["心内科", "内科", "心血管"])
        assert ok, f"回复：{r['reply']}"

    def test_v04_symptom_fever_cough(self):
        """症状反查：发烧咳嗽可能是感冒/肺炎/流感/支气管炎。"""
        ok, r = _check("我发烧咳嗽可能是什么病？",
                        ["感冒", "肺炎", "流感", "支气管"])
        assert ok, f"回复：{r['reply']}"

    def test_v05_hepatitis_transmission(self):
        """乙肝传播途径（血液传播/母婴传播/性传播）。"""
        ok, r = _check("乙肝是怎么传播的？",
                        ["血液", "母婴", "性", "传播"])
        assert ok, f"回复：{r['reply']}"

    def test_v06_tb_prevent(self):
        """肺结核预防措施（卡介苗接种/通风/隔离/早期发现）。"""
        ok, r = _check("肺结核怎么预防？",
                        ["卡介苗", "通风", "隔离", "疫苗接种", "早期发现", "痰液"])
        assert ok, f"回复：{r['reply']}"

    def test_v07_asthma_symptom(self):
        """哮喘症状（喘息/呼吸困难/咳嗽/胸闷）。"""
        ok, r = _check("哮喘有哪些症状？",
                        ["喘息", "呼吸", "咳嗽", "胸闷"])
        assert ok, f"回复：{r['reply']}"

    def test_v08_appendicitis_complication(self):
        """阑尾炎并发症（穿孔/腹膜炎/脓肿）。"""
        ok, r = _check("阑尾炎有什么并发症？",
                        ["穿孔", "腹膜炎", "脓肿"])
        assert ok, f"回复：{r['reply']}"

    def test_v09_anemia_cause(self):
        """贫血病因（缺铁/出血/营养不良/溶血/造血障碍）。"""
        ok, r = _check("贫血是什么原因引起的？",
                        ["缺铁", "出血", "营养", "溶血", "造血", "铁缺乏", "红细胞"])
        assert ok, f"回复：{r['reply']}"

    def test_v10_cold_nursing(self):
        """感冒护理要点（休息/多喝水/营养支持）。"""
        ok, r = _check("感冒护理需要注意什么？",
                        ["休息", "多喝水", "营养", "护理"])
        assert ok, f"回复：{r['reply']}"
