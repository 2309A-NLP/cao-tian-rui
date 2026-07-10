"""
工单11 医疗挂号Agent - 结构化日志
日志文件：logs/agent_YYYY-MM-DD.jsonl  （每行一个 JSON 事件）

设计要点：
  - 每行一个完整 JSON 对象（JSONL 格式），方便 grep/jq 查询
  - 按日期自动滚动文件，避免单文件过大
  - 同时输出到 stderr（开发调试用），ERROR/WARNING 级别总是打印
"""
import json        # 标准库：JSON 序列化/反序列化
import logging     # 标准库：Python 内置日志框架（此处只用于级别常量）
import os          # 标准库：读取环境变量 LOG_LEVEL
import traceback   # 标准库：格式化异常堆栈信息
import uuid        # 标准库：生成全局唯一 ID（UUID），用于 trace_id
from datetime import datetime   # 标准库：获取当前时间戳
from pathlib import Path        # 标准库：路径操作

# 日志目录：与 src/ 同级的 logs/ 文件夹
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)   # 目录不存在时自动创建，已存在不报错

# 日志级别从环境变量读取，默认 INFO；DEBUG 模式下会打印更多信息
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _log_file() -> Path:
    """
    返回今天对应的日志文件路径（按日期自动滚动）。
    例如：logs/agent_2026-07-06.jsonl
    每天的日志写入独立文件，方便按日期归档和清理。
    """
    return LOG_DIR / f"agent_{datetime.now().strftime('%Y-%m-%d')}.jsonl"


class _DateEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，扩展标准 JSONEncoder 以处理日期类型。

    问题背景：
      MySQL DictCursor 查询结果中日期字段（如 sch_date）会返回 Python
      date/datetime 对象，而标准 json.dumps() 不知道如何序列化它们，
      会抛出 TypeError。

    解决方案：
      重写 default() 方法，遇到 date/datetime 对象时转为 ISO 格式字符串
      "YYYY-MM-DD" 或 "YYYY-MM-DDTHH:MM:SS"。
    """
    def default(self, obj):
        from datetime import date, datetime  # 局部导入，避免与模块顶层同名变量冲突
        if isinstance(obj, (date, datetime)):  # 若对象是 date 或 datetime 类型
            return obj.isoformat()              # 转为 ISO 8601 字符串
        return super().default(obj)            # 其他类型交给父类处理（可能抛 TypeError）


def _write(event: dict):
    """
    将事件字典序列化为 JSON 并追加写入今天的日志文件。
    同时根据日志级别决定是否输出到控制台（stderr）。
    """
    # ensure_ascii=False：允许中文直接写入，不转义为 \uXXXX
    line = json.dumps(event, ensure_ascii=False, cls=_DateEncoder)
    with open(_log_file(), "a", encoding="utf-8") as f:
        f.write(line + "\n")   # 每条日志占一行（JSONL 格式）
    # DEBUG 模式下打印所有日志；非 DEBUG 模式只打印 ERROR/WARNING/CRITICAL
    if _LOG_LEVEL == "DEBUG" or event.get("level") in ("ERROR", "WARNING", "CRITICAL"):
        print(f"[{event['level']}] {event.get('message', '')} | trace={event.get('trace_id', '-')}")


def new_trace() -> str:
    """
    生成一次请求/会话的唯一 trace_id，用于在日志中串联同一请求的所有事件。
    使用 UUID4 的前 12 个十六进制字符（48 bit 随机性，冲突概率极低）。
    """
    return uuid.uuid4().hex[:12]


def log_info(message: str, trace_id: str = "-", **kwargs):
    """
    记录 INFO 级别日志。
    **kwargs 允许传入任意附加字段（如 user_id、query），
    它们会被合并到 JSON 事件对象中，方便结构化查询。
    """
    _write({"ts": datetime.now().isoformat(), "level": "INFO",
            "trace_id": trace_id, "message": message, **kwargs})


def log_debug(message: str, trace_id: str = "-", **kwargs):
    """
    记录 DEBUG 级别日志，仅在 LOG_LEVEL=DEBUG 时实际写入，
    避免生产环境日志量过大。
    """
    if _LOG_LEVEL == "DEBUG":  # 非 DEBUG 模式直接跳过，不写文件
        _write({"ts": datetime.now().isoformat(), "level": "DEBUG",
                "trace_id": trace_id, "message": message, **kwargs})


def log_warning(message: str, trace_id: str = "-", **kwargs):
    """记录 WARNING 级别日志（可恢复的异常，如 LLM 降级、解析失败）。"""
    _write({"ts": datetime.now().isoformat(), "level": "WARNING",
            "trace_id": trace_id, "message": message, **kwargs})


def log_error(message: str, trace_id: str = "-", exc: Exception = None, **kwargs):
    """
    记录 ERROR 级别日志。
    若传入 exc（异常对象），额外记录异常类型名和完整堆栈，方便排查问题。
    """
    payload = {"ts": datetime.now().isoformat(), "level": "ERROR",
               "trace_id": trace_id, "message": message, **kwargs}
    if exc:
        payload["exception"] = type(exc).__name__          # 异常类名（如 pymysql.OperationalError）
        payload["traceback"] = traceback.format_exc()       # 完整堆栈字符串
    _write(payload)


def log_tool_call(tool: str, params: dict, result, elapsed_ms: float,
                  trace_id: str = "-", error: str = None):
    """
    记录一次工具函数调用的完整信息（入参、出参、耗时）。
    用于分析 Agent 的工具使用情况和性能瓶颈。

    参数：
      tool       : 工具函数名（如 "query_schedule"）
      params     : 调用参数字典
      result     : 工具返回值（正常时）
      elapsed_ms : 执行耗时（毫秒）
      error      : 错误描述字符串（出错时填写，正常时为 None）
    """
    _write({
        "ts": datetime.now().isoformat(),
        "level": "INFO" if not error else "ERROR",   # 有错误时升级为 ERROR 级别
        "trace_id": trace_id,
        "message": f"tool_call:{tool}",               # 固定前缀便于日志过滤
        "tool": tool,
        "params": params,
        "result": result if not error else None,       # 出错时不记录 result
        "error": error,
        "elapsed_ms": round(elapsed_ms, 2),            # 保留 2 位小数
    })


def log_llm_call(phase: str, messages: list, response: str, elapsed_ms: float,
                 trace_id: str = "-", error: str = None):
    """
    记录一次 LLM API 调用的完整信息（输入消息列表、输出、耗时）。
    用于分析 token 使用量和 LLM 响应质量。

    参数：
      phase      : 调用阶段标签（如 "attempt1"、"fallback"）
      messages   : 发送给 LLM 的消息列表（包含 system/user/assistant/tool 角色）
      response   : LLM 返回的文本内容
      elapsed_ms : 调用耗时（毫秒）
    """
    _write({
        "ts": datetime.now().isoformat(),
        "level": "INFO" if not error else "ERROR",
        "trace_id": trace_id,
        "message": f"llm_call:{phase}",
        "phase": phase,
        "messages": messages,    # 完整上下文，方便回放
        "response": response,
        "error": error,
        "elapsed_ms": round(elapsed_ms, 2),
    })


def log_rollback(tool: str, reason: str, trace_id: str = "-", **kwargs):
    """
    记录数据库事务回滚事件（WARNING 级别）。
    当 book_appointment / cancel_appointment 因校验失败或异常回滚时调用。
    """
    _write({
        "ts": datetime.now().isoformat(),
        "level": "WARNING",
        "trace_id": trace_id,
        "message": f"rollback:{tool}",   # 固定前缀便于过滤
        "tool": tool,
        "reason": reason,
        **kwargs,
    })
