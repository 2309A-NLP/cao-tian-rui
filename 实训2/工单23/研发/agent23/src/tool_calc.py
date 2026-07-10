"""
Calculator 工具：处理单位换算和数学计算类题目。

特点：
- 优先匹配单位换算规则（如 "100 km to mile"），无需 LLM
- 回退到 asteval 安全计算器处理数学表达式（如 "3.14 * 2 ** 2"）
- 使用 asteval 而非 Python 原生 eval，防止代码注入攻击

安全说明：
  Python 内置 eval() 可以执行任意代码，存在安全风险。
  asteval（第三方包）是一个受限的表达式求值器，只支持数学运算，
  不能访问文件系统、导入模块或执行系统命令。
"""
import logging    # 标准库：日志记录
import re         # 标准库：正则表达式，用于匹配单位换算规则
import threading  # 标准库：线程工具，用于线程本地存储
from dataclasses import dataclass  # 标准库：数据类

# asteval：第三方包，安全的数学表达式求值器
# - 支持算术运算（+/-/*//**/%）、数学函数（sin/cos/sqrt 等）
# - 不支持 import、exec、__class__ 等危险操作
# - 比 sympy 轻量，比原生 eval 安全
from asteval import Interpreter

logger = logging.getLogger(__name__)

# 线程本地存储：每个线程有自己独立的 Interpreter 实例
# 原因：Interpreter 不是线程安全的，多线程共用会有状态混乱问题
_aeval_local = threading.local()


def _get_aeval() -> Interpreter:
    """
    获取当前线程的 asteval Interpreter 实例。
    首次调用时创建，后续复用（避免重复初始化开销）。

    :return: 当前线程独有的 Interpreter 实例
    """
    if not hasattr(_aeval_local, "interp"):
        _aeval_local.interp = Interpreter()  # 首次调用时创建
    return _aeval_local.interp  # 返回已有实例


# 单位换算规则表（可扩展）
# 每条规则格式：(正则模式, 转换函数 lambda, 输出单位字符串)
# 正则中的第一个捕获组匹配数值部分
_UNIT_RULES: list[tuple] = [
    # ── 长度换算 ──────────────────────────────────────────────────────────────
    # 千米 → 英里（1 km = 0.621371 mile）
    (r"(\d+\.?\d*)\s*km\s+to\s+mile",    lambda v: v * 0.621371,  "mile"),
    # 英里 → 千米（1 mile = 1.60934 km）
    (r"(\d+\.?\d*)\s*mile\s+to\s+km",    lambda v: v * 1.60934,   "km"),
    # 米 → 英尺（1 m = 3.28084 ft）
    (r"(\d+\.?\d*)\s*m\s+to\s+ft",       lambda v: v * 3.28084,   "ft"),
    # 英尺 → 米（1 ft = 0.3048 m）
    (r"(\d+\.?\d*)\s*ft\s+to\s+m",       lambda v: v * 0.3048,    "m"),

    # ── 温度换算 ──────────────────────────────────────────────────────────────
    # 摄氏度 → 华氏度（°F = °C × 9/5 + 32）
    (r"(\d+\.?\d*)\s*[Cc]\s+to\s+[Ff]",  lambda v: v * 9/5 + 32,  "°F"),
    # 华氏度 → 摄氏度（°C = (°F - 32) × 5/9）
    (r"(\d+\.?\d*)\s*[Ff]\s+to\s+[Cc]",  lambda v: (v - 32) * 5/9, "°C"),

    # ── 重量换算 ──────────────────────────────────────────────────────────────
    # 千克 → 磅（1 kg = 2.20462 lb）
    (r"(\d+\.?\d*)\s*kg\s+to\s+lb",      lambda v: v * 2.20462,   "lb"),
    # 磅 → 千克（1 lb = 0.453592 kg）
    (r"(\d+\.?\d*)\s*lb\s+to\s+kg",      lambda v: v * 0.453592,  "kg"),

    # ── 面积换算 ──────────────────────────────────────────────────────────────
    # 平方千米 → 平方英里（1 km² = 0.386102 square miles）
    (r"(\d+\.?\d*)\s*km2\s+to\s+mile2",  lambda v: v * 0.386102,  "square miles"),
]


@dataclass
class CalcResult:
    """
    计算结果数据类。

    Attributes:
        value:      计算结果（数字或字符串）
        unit:       结果单位（如 "km"、"°F"），无单位时为空字符串
        expression: 原始输入表达式（用于调试日志）
        success:    计算是否成功
    """
    value: float | str   # 计算结果值
    unit: str            # 结果单位（可能为空）
    expression: str      # 原始表达式
    success: bool        # 是否成功


def calculate(expression: str) -> CalcResult:
    """
    计算数学表达式或单位换算。

    处理顺序：
    1. 优先匹配单位换算规则（正则 + lambda，速度最快）
    2. 不匹配时回退到 asteval 数学表达式求值

    :param expression: 输入表达式，例如：
                       - "3.14 * 2 ** 2"    → 数学计算
                       - "100 km to mile"   → 单位换算
                       - "32 F to C"        → 温度换算
    :return:           CalcResult 数据类实例
    """
    expr = expression.strip()  # 去首尾空格

    # ── 第1步：尝试单位换算规则 ────────────────────────────────────────────────
    for pattern, convert_fn, out_unit in _UNIT_RULES:
        m = re.search(pattern, expr, re.IGNORECASE)  # 大小写不敏感匹配
        if m:
            val = float(m.group(1))       # 提取数值并转为浮点数
            result = convert_fn(val)       # 调用对应换算公式
            result = round(result, 6)      # 保留6位小数，避免浮点精度问题
            logger.debug("Unit convert: %r → %s %s", expr, result, out_unit)
            return CalcResult(value=result, unit=out_unit, expression=expr, success=True)

    # ── 第2步：回退到 asteval 数学表达式求值 ───────────────────────────────────
    try:
        aeval = _get_aeval()           # 获取当前线程的 Interpreter 实例
        result = aeval(expr)           # 执行表达式求值

        if aeval.error:
            # asteval 遇到错误时不抛异常，而是将错误存储在 aeval.error 列表中
            raise ValueError(aeval.error[0].get_error())  # 主动抛出让 except 捕获

        result = float(result)         # 将结果转为浮点数（asteval 可能返回 int）
        logger.debug("Math eval: %r → %s", expr, result)
        return CalcResult(value=result, unit="", expression=expr, success=True)

    except Exception as e:
        # 单位换算和数学求值都失败，记录警告并返回失败结果
        logger.warning("Calc failed [%r]: %s", expr, e)
        return CalcResult(value=0.0, unit="", expression=expr, success=False)
