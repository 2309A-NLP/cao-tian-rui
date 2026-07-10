"""
工单11 医疗挂号Agent - FastAPI 接口
端口: 8011

接口列表:
  GET  /health                    健康检查
  POST /chat                      Agent 对话（挂号/取消/查询全走这里）
  GET  /slots                     号源查询（前端筛选表格直接调用）
  GET  /appointments/{user_id}    我的预约列表 + 统计
  GET  /schedule                  医生排班查询

依赖的第三方库说明：
  - fastapi   : 高性能 Python Web 框架，基于 Starlette + Pydantic，支持异步
  - pydantic  : 数据校验和序列化库，FastAPI 用它定义请求/响应模型（BaseModel）
"""
import os   # 标准库：读取 CORS_ORIGINS 等环境变量
from datetime import date, timedelta   # 标准库：日期操作
from typing import Optional            # 标准库：Optional 类型注解

# FastAPI：Python 高性能 Web 框架
# Depends：依赖注入，用于在路由处理前执行鉴权等中间件逻辑
# FastAPI：应用实例类
# HTTPException：抛出 HTTP 错误响应
# Query：声明查询参数（URL ?key=value 格式）
from fastapi import Depends, FastAPI, HTTPException, Query
# CORSMiddleware：跨域资源共享中间件，允许前端页面（不同端口）调用本服务
from fastapi.middleware.cors import CORSMiddleware
# BaseModel：Pydantic 的基类，用于定义请求体和响应体的数据结构
from pydantic import BaseModel

from src.agent import run_agent       # Agent 主调度函数
from src import logger                # 结构化日志
from src.auth import verify_api_key   # API Key 鉴权依赖
from src import tools                 # 工具函数统一入口（向后兼容）
from src.config import VALID_TITLES, MAX_FUTURE_DAYS   # 业务常量

# 创建 FastAPI 应用实例，设置 API 文档标题和版本（访问 /docs 可查看 Swagger UI）
app = FastAPI(title="医疗挂号Agent", version="2.0")

# ── 从环境变量读取允许跨域的源（Origin）列表 ──
# CORS（跨域资源共享）：浏览器的安全策略，默认禁止跨域请求。
# 需要在服务器端声明哪些来源（Origin）可以访问 API。
_origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    # 默认允许本机的常见开发端口（8011=本服务，8000=网关，3000=前端）
    "http://localhost:8011,http://127.0.0.1:8011,"
    "http://localhost:8000,http://127.0.0.1:8000,"    # 门户统一入口
    "http://localhost:3000,http://127.0.0.1:3000",
    # "null" 已移除：file:// 的 origin=null 会允许任意本地 HTML 跨域调用所有接口（安全风险）
).split(",")]

# 注册 CORS 中间件，配置允许的来源/方法/请求头
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,                                    # 允许的来源列表
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],        # 允许的 HTTP 方法
    allow_headers=["*"],                                       # 允许所有请求头（含自定义的 X-API-Key）
)


# ──────────────── 请求/响应模型 ────────────────

class ChatRequest(BaseModel):
    """
    POST /chat 的请求体模型。
    Pydantic 的 BaseModel 会自动解析、校验 JSON 请求体。

    字段：
      user_id : 已登录用户 ID（整数）
      query   : 用户自然语言输入（字符串）
    """
    user_id: int
    query: str

class ChatResponse(BaseModel):
    """
    POST /chat 的响应体模型，控制返回给前端的字段。

    字段：
      reply    : Agent 的自然语言回复文本
      trace_id : 本次请求的链路追踪 ID（方便查日志定位问题）
      trace    : Agent 调用工具的步骤列表（前端可展示思考过程）
    """
    reply: str
    trace_id: str
    trace: list = []   # 默认空列表，未出现工具调用时为空


# ──────────────── 健康检查 ────────────────

@app.get("/health")
def health():
    """
    GET /health：健康检查接口，供 K8s/Nginx 等探活使用。
    不需要鉴权，返回固定 JSON 表示服务运行正常。
    """
    return {"status": "ok", "service": "registration-agent"}


# ──────────────── Agent 对话（核心） ────────────────

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(req: ChatRequest):
    """
    POST /chat：核心对话接口，接收用户自然语言输入，返回 Agent 回复。

    流程：
      1. 生成 trace_id（用于日志关联）
      2. 调用 run_agent() 驱动 LLM 多轮工具循环
      3. 返回回复文本 + trace_id + 工具调用步骤

    鉴权：依赖 verify_api_key（请求头需携带 X-API-Key）
    """
    trace_id = logger.new_trace()   # 生成本次请求的唯一追踪 ID
    result = run_agent(req.user_id, req.query, trace_id=trace_id)   # 调用 Agent 主函数
    return ChatResponse(reply=result["reply"], trace_id=trace_id, trace=result["trace"])


# ──────────────── REST: 号源查询（前端表格） ────────────────

@app.get("/slots", dependencies=[Depends(verify_api_key)])
def get_slots(
    dept:       Optional[str] = Query(None, description="科室，如 儿科"),
    date_start: Optional[str] = Query(None, description="开始日期 YYYY-MM-DD"),
    date_end:   Optional[str] = Query(None, description="结束日期 YYYY-MM-DD"),
    slot_type:  Optional[str] = Query(None, description="号源类型：专家|普通"),
    doctor:     Optional[str] = Query(None, description="医生姓名（模糊）"),
):
    """
    GET /slots：前端筛选栏专用号源查询接口。
    与 /chat 不同，此接口直接返回结构化数据，不经过 LLM。

    默认行为：
      - 不指定日期：默认今天起 MAX_FUTURE_DAYS 天
      - 不指定科室：返回所有科室（最多 50 条）
      - 过滤今日已结束时段（与 tools.query_schedule 行为保持一致）
    """
    today = date.today()
    max_end = (today + timedelta(days=MAX_FUTURE_DAYS)).isoformat()   # 最大允许结束日期

    # ── 日期格式校验（防止 SQL 注入和运行时错误）──
    for label, val in [("date_start", date_start), ("date_end", date_end)]:
        if val:
            try:
                date.fromisoformat(val)   # 尝试将字符串解析为 date，格式不对会抛 ValueError
            except ValueError:
                raise HTTPException(status_code=422, detail=f"{label} 格式无效，需 YYYY-MM-DD")

    ds = date_start or today.isoformat()                                    # 开始日期：未传则今天
    de = min(date_end or (today + timedelta(days=MAX_FUTURE_DAYS)).isoformat(), max_end)  # 截断到最大

    # ── slot_type 白名单校验 ──
    if slot_type and slot_type not in VALID_TITLES:
        raise HTTPException(status_code=422, detail=f"slot_type 无效，可选: {VALID_TITLES}")

    if not dept:
        # ── 未指定科室：直接查数据库，返回所有科室（最多50条）──
        from src.database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        # 联表查询：schedule → doctor → department，只取有余量的行
        sql = """
            SELECT s.sch_id, d.d_Name AS doctor_name, d.d_Profession,
                   dep.dep_Name AS dept,
                   s.sch_type AS slot_type,
                   s.sch_date AS date,
                   s.sch_time_slot AS time_slot,
                   s.sch_available AS available,
                   s.sch_total AS total
            FROM schedule s
            JOIN doctor d ON d.d_ID = s.d_id
            JOIN department dep ON dep.dep_ID = d.dep_ID
            WHERE s.sch_date BETWEEN %s AND %s
              AND s.sch_available > 0
        """
        args = [ds, de]
        if slot_type:
            sql += " AND s.sch_type = %s"; args.append(slot_type)   # 按号源类型过滤
        if doctor:
            sql += " AND d.d_Name LIKE %s"; args.append(f"%{doctor}%")  # 医生姓名模糊匹配
        sql += " ORDER BY s.sch_date, dep.dep_Name, s.sch_time_slot LIMIT 50"  # 排序并限制条数
        cur.execute(sql, args)
        rows = cur.fetchall()
        conn.close()

        # ── 后过滤：去掉今天已结束的时间段 ──
        from datetime import datetime as _dt
        _now = _dt.now()
        _today_str = today.isoformat()
        _slot_end_hour = {"09:00-11:00": 11, "14:00-17:00": 17, "19:00-21:00": 21}
        rows = [
            r for r in rows
            # 非今天不过滤；今天只保留还未结束的时间段
            if r["date"].isoformat() != _today_str
            or _now.hour < _slot_end_hour.get(r["time_slot"], 24)
        ]

        # MySQL DictCursor 返回的 date 对象需要转字符串，才能被 JSON 序列化
        data = [{**r, "date": r["date"].isoformat()} for r in rows]
        return {"ok": True, "slots": data, "total": len(data)}

    # ── 指定了科室：走 tools.query_schedule 函数（含完整的过滤逻辑）──
    result = tools.query_schedule(
        dept=dept, date_start=ds, date_end=de,
        title=slot_type, doctor_name=doctor,
    )
    if not result["ok"]:
        return {"ok": False, "error": result["error"], "slots": []}

    # 同样转换 date 对象为字符串
    data = [
        {**s, "date": s["date"].isoformat() if hasattr(s["date"], "isoformat") else s["date"]}
        for s in result["schedules"]
    ]
    return {"ok": True, "slots": data, "total": len(data)}


# ──────────────── REST: 我的预约 ────────────────

@app.get("/appointments/{user_id}", dependencies=[Depends(verify_api_key)])
def get_appointments(
    user_id: int,
    dept:   Optional[str] = Query(None),
    status: Optional[int] = Query(None, description="1=正常 0=已取消"),
    bucket: Optional[str] = Query(None, description="upcoming=待就诊 / history=历史(已过期或已取消)"),
):
    """
    GET /appointments/{user_id}：返回该用户所有挂号记录及统计数据。

    路径参数：
      user_id : 用户 ID

    查询参数：
      dept   : 按科室过滤（可选）
      status : 按挂号状态过滤，1=正常/待就诊，0=已取消（可选）
      bucket : upcoming=只看未过期的有效预约；history=只看已过期或已取消的记录

    每条记录附带 bucket 字段，前端无需自己算：
      "upcoming" : reg_status=1 且就诊日 > 今天，或就诊日=今天且时段未结束
      "history"  : reg_status=0（已取消）或就诊日已过

    返回：
      {
        "ok": True,
        "appointments": [{... , "bucket": "upcoming|history"}, ...],
        "stats": {"pending": n, "total": n, "canceled": n}
      }
    """
    result = tools.get_user_history(user_id=user_id, dept=dept)
    if not result["ok"]:
        return {"ok": False, "error": result["error"], "appointments": [], "stats": {}}

    history = result["history"]
    today   = date.today()

    # 时段结束时间映射（用于判断今日号是否已过场）
    from datetime import datetime as _dt
    _now_hour = _dt.now().hour
    _slot_end_hour = {"09:00-11:00": 11, "14:00-17:00": 17, "19:00-21:00": 21}

    def _classify(r):
        """返回 upcoming / history。"""
        if r["reg_status"] == 0:
            return "history"   # 已取消
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(str(r["date"]))
        if d < today:
            return "history"   # 就诊日已过
        if d == today:
            # 今天：看时段是否已过
            end_h = _slot_end_hour.get(r.get("time_slot"), 24)
            return "history" if _now_hour >= end_h else "upcoming"
        return "upcoming"       # 未来日期

    # ── 构建过滤后的记录列表 ──
    records = []
    for r in history:
        if status is not None and r["reg_status"] != status:
            continue
        b = _classify(r)
        if bucket is not None and b != bucket:
            continue
        d = dict(r)
        d["date"] = d["date"].isoformat() if hasattr(d["date"], "isoformat") else d["date"]
        # reg_time 也序列化（DATETIME 对象 → 字符串）
        if d.get("reg_time") is not None and hasattr(d["reg_time"], "isoformat"):
            d["reg_time"] = d["reg_time"].isoformat(sep=" ")
        d["bucket"] = b
        records.append(d)

    # ── 统计数据基于全量 history，不受 bucket/status 过滤影响 ──
    pending  = sum(1 for r in history if _classify(r) == "upcoming")
    canceled = sum(1 for r in history if r["reg_status"] == 0)

    return {
        "ok": True,
        "appointments": records,
        "stats": {"pending": pending, "total": len(history), "canceled": canceled},
    }


# ──────────────── REST: 医生排班 ────────────────

@app.get("/schedule", dependencies=[Depends(verify_api_key)])
def get_doctor_schedule(
    doctor:     str           = Query(..., description="医生姓名（支持模糊）"),
    date_start: Optional[str] = Query(None),
    date_end:   Optional[str] = Query(None),
):
    """
    GET /schedule：查询指定医生的排班时间表（含已满号源）。

    查询参数：
      doctor     : 医生姓名（必填，支持模糊匹配，如"张"匹配"张伟"）
      date_start : 开始日期（可选，默认今天）
      date_end   : 结束日期（可选，默认今天起 13 天，即两周）

    ... 代表必填参数（FastAPI Query 的 ... 符号表示必须传入，不可省略）
    """
    today = date.today()
    # ── 日期格式校验 ──
    for label, val in [("date_start", date_start), ("date_end", date_end)]:
        if val:
            try:
                date.fromisoformat(val)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"{label} 格式无效，需 YYYY-MM-DD")
    ds = date_start or today.isoformat()                          # 默认今天
    de = date_end   or (today + timedelta(days=13)).isoformat()   # 默认两周内

    result = tools.get_doctor_schedule(doctor_name=doctor, date_start=ds, date_end=de)
    if not result["ok"]:
        return {"ok": False, "error": result["error"], "schedules": []}

    # 将 date 对象转为 ISO 字符串，保证 JSON 可序列化
    data = [
        {**s, "date": s["date"].isoformat() if hasattr(s["date"], "isoformat") else s["date"]}
        for s in result["schedules"]
    ]
    return {"ok": True, "doctor": result["doctor"], "schedules": data}
