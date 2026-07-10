"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-MCP

amap_mcp —— 高德地图 MCP Server（业务组合层）
--------------------------------------------
对接高德 Web 服务 REST API，将"地理编码 → 周边搜索/路线规划"等多步操作
组合为面向医疗场景的业务化 tool。

传输方式：stdio

设计说明：
  不做高德官方 SSE MCP 的重复封装，而是把高德的原子 API（地点搜索/周边搜索/路线规划）
  按医疗场景组合。例如 nearby_hotels(hospital="北京协和医院") 内部会：
    1. 用 place/text 搜索医院得到坐标
    2. 用 place/around 以该坐标为中心搜索酒店
  这样路由 Agent 只需理解业务，不用组合原子 API。

Tools:
  - hospital_search       医院搜索（返回地址、坐标、详情）
  - route_planning        路线规划（驾车/公交/步行/骑行）
  - nearby_search         通用周边搜索（自定义关键词）
  - nearby_hotels         医院附近酒店（默认 1km）
  - nearby_restaurants    医院附近餐饮��默认 500m）
  - nearby_pharmacies     医院附近药店（默认 1km）
  - nearby_parking        医院附近停车场（默认 500m）
"""
import asyncio  # 标准库：异步并发，用于 asyncio.gather 并发地理编码请求
import os  # 标准库：读取环境变量 AMAP_API_KEY、MCP_UPSTREAM_TIMEOUT
import sys  # 标准库：sys.stderr 输出启动日志
from typing import Any, Optional  # 标准库：类型注解

# httpx 包：异步 HTTP 客户端，替代 requests。
# 用于调用高德 REST API（httpx.AsyncClient 支持 async/await）。
import httpx
from dotenv import load_dotenv  # python-dotenv：加载 .env 文件
# mcp.server.fastmcp：MCP SDK 提供的高级 Server 封装
# FastMCP 用 @mcp.tool() 装饰器将 Python 函数注册为 MCP tool，并自动生成 JSON Schema
from mcp.server.fastmcp import FastMCP

# 解析 04-MCP/ 根目录（本文件位于 mcp_servers/，上一级是 04-MCP/）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))  # 加载环境变量

AMAP_KEY = os.getenv("AMAP_API_KEY", "")  # 高德 Web 服务 API Key
AMAP_BASE = "https://restapi.amap.com"     # 高德 REST API 基础 URL
TIMEOUT = float(os.getenv("MCP_UPSTREAM_TIMEOUT", "10.0"))  # 每次高德 API 调用的超时秒数

# 创建名为 "amap" 的 MCP Server 实例
mcp = FastMCP("amap")

# 进程级地理编码缓存：(name, city) -> poi dict
# 热门医院名在同一 MCP Server 进程内只查一次，之后 nearby_* 直接用缓存
_GEOCODE_CACHE: dict[tuple[str, str], dict] = {}


# ────────── 内部辅助 ──────────

async def _geocode(client: httpx.AsyncClient, name: str, city: str = "北京") -> Optional[dict]:
    """
    通过高德 place/text 接口关键词搜索，获取地点的第一个结果（含坐标）。
    命中本地缓存时跳过 API 调用（避免重复请求）。

    Args:
        client: 已创建的 httpx.AsyncClient 实例（复用连接）
        name: 地点名称，如"北京协和医院"
        city: 城市名，默认"北京"

    Returns:
        POI 字典（含 name/address/location/tel 等字段）；未找到时返回 None
    """
    key = (name, city)
    # 命中缓存：直接返回缓存的 POI 数据，跳过 API 调用
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    # 调用高德 place/text 接口搜索关键词
    r = await client.get(f"{AMAP_BASE}/v3/place/text", params={
        "key": AMAP_KEY,         # API Key
        "keywords": name,        # 搜索关键词
        "city": city,            # 限定城市
        "offset": 1,             # 只取第一个结果（减少传输量）
        "extensions": "base",    # 返回基础字段（不返回扩展详情）
    })
    r.raise_for_status()  # HTTP 非 200 时抛出异常
    data = r.json()
    # status != "1" 表示高德 API 返回错误，pois 为空表示未找到
    if data.get("status") != "1" or not data.get("pois"):
        return None
    poi = data["pois"][0]  # 取第一个搜索结果
    _GEOCODE_CACHE[key] = poi  # 写入缓存
    return poi


async def _around(client: httpx.AsyncClient, location: str, keywords: str,
                  radius: int, types: str = "", offset: int = 10) -> list[dict]:
    """
    高德 place/around 周边搜索，以指定坐标为中心按距离排序返回 POI 列表。

    Args:
        client: httpx.AsyncClient 实例
        location: 中心坐标，格式"经度,纬度"（如"116.397428,39.90923"）
        keywords: 搜索关键词（如"酒店|宾馆"）
        radius: 搜索半径（米）
        types: 高德 POI 类型代码（可选），留空则按关键词搜索
        offset: 返回条数，默认 10

    Returns:
        POI 列表（可能为空列表）
    """
    r = await client.get(f"{AMAP_BASE}/v3/place/around", params={
        "key": AMAP_KEY,
        "location": location,   # 中心坐标
        "keywords": keywords,   # 搜索关键词
        "types": types,         # POI 类型
        "radius": radius,       # 半径（米）
        "offset": offset,       # 返回条数
        "sortrule": "distance", # 按距离从近到远排序
        "extensions": "base",   # 只返回基础字段
    })
    r.raise_for_status()
    data = r.json()
    # status != "1" 表示 API 异常，返回空列表
    if data.get("status") != "1":
        return []
    return data.get("pois", [])  # 返回 POI 列表（可能为空）


def _fmt_poi(p: dict) -> dict:
    """
    裁剪 POI 字段，只保留业务用得到的核心字段。

    Args:
        p: 高德 API 返回的原始 POI 字典

    Returns:
        精简后的 POI 字典（name/address/location/distance/tel/type）
    """
    return {
        "name": p.get("name"),        # 地点名称
        "address": p.get("address"),  # 详细地址
        "location": p.get("location"),  # 坐标字符串"lng,lat"
        # distance 字段在周边搜索时有值（米），需转 int；place/text 搜索无此字段
        "distance": int(p.get("distance", 0)) if str(p.get("distance", "")).isdigit() else None,
        "tel": p.get("tel"),          # 电话
        "type": p.get("type"),        # POI 类型描述
    }


def _check_key():
    """
    检查 AMAP_API_KEY 是否已配置。

    Returns:
        None 表示已配置（可以继续）；dict 表示未配置（作为工具返回的错误信息）
    """
    if not AMAP_KEY:
        return {"ok": False, "error": "AMAP_API_KEY 未配置，请在 .env 填入"}
    return None


# ────────── Tools ──────────

@mcp.tool()
async def hospital_search(name: str, city: str = "北京") -> dict[str, Any]:
    """
    搜索医院，返回位置、地址、电话。

    Args:
        name: 医院名，如"北京协和医院"、"儿童医院"
        city: 城市，默认"北京"

    Returns:
        { "ok": bool, "hospital": {name, address, location, tel, ...} }
    """
    # 先检查 API Key 是否配置
    if err := _check_key():
        return err
    try:
        # 创建异步 HTTP 客户端（with 块结束自动关闭）
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            poi = await _geocode(client, name, city)  # 搜索医院坐标
            if not poi:
                # 高德未找到该医院
                return {"ok": False, "error": f"未找到医院: {name}"}
            return {"ok": True, "hospital": _fmt_poi(poi)}
    except httpx.RequestError as e:
        # 网络连接失败（DNS/超时/连接拒绝）
        return {"ok": False, "error": f"高德 API 连接失败: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def route_planning(
    origin: str,
    destination: str,
    mode: str = "driving",
    city: str = "北京",
) -> dict[str, Any]:
    """
    路线规划。支持地名或"经度,纬度"格式。

    Args:
        origin: 起点，如"北京南站" 或 "116.397428,39.90923"
        destination: 终点，如"北京协和医院"
        mode: driving | walking | bicycling | transit
        city: 公交查询必需的城市，默认"北京"

    Returns:
        { "ok": bool, "mode": str, "distance_m": int, "duration_s": int, "steps_summary": [...] }
    """
    if err := _check_key():
        return err
    # 校验出行方式是否合法
    if mode not in ("driving", "walking", "bicycling", "transit"):
        return {"ok": False, "error": f"mode 无效，可选: driving/walking/bicycling/transit"}

    def _is_coord(s: str) -> bool:
        """判断字符串是否为"lng,lat"坐标格式。"""
        parts = s.split(",")
        if len(parts) != 2:
            return False  # 不是逗号分隔的两段
        try:
            float(parts[0]); float(parts[1])  # 两段都能转 float
            return True
        except ValueError:
            return False  # 包含非数字字符，是地名

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 起终点转坐标，若两端都是地名则并发 geocode，节省约 500ms
            origin_is_coord = _is_coord(origin)
            dest_is_coord = _is_coord(destination)

            if not origin_is_coord and not dest_is_coord:
                # 两端都是地名：并发发起两个 geocode 请求
                o_task = _geocode(client, origin, city)
                d_task = _geocode(client, destination, city)
                o, d = await asyncio.gather(o_task, d_task)  # 并发等待
                if not o:
                    return {"ok": False, "error": f"起点未找到: {origin}"}
                if not d:
                    return {"ok": False, "error": f"终点未找到: {destination}"}
                origin, destination = o["location"], d["location"]  # 替换为坐标
            elif not origin_is_coord:
                # 只有起点是地名
                o = await _geocode(client, origin, city)
                if not o:
                    return {"ok": False, "error": f"起点未找到: {origin}"}
                origin = o["location"]
            elif not dest_is_coord:
                # 只有终点是地名
                d = await _geocode(client, destination, city)
                if not d:
                    return {"ok": False, "error": f"终点未找到: {destination}"}
                destination = d["location"]

            # 根据出行方式选择不同的高德路线规划接口
            if mode == "driving":
                # 驾车规划（v3）
                url = f"{AMAP_BASE}/v3/direction/driving"
                params = {"key": AMAP_KEY, "origin": origin, "destination": destination}
            elif mode == "walking":
                # 步行规划（v3）
                url = f"{AMAP_BASE}/v3/direction/walking"
                params = {"key": AMAP_KEY, "origin": origin, "destination": destination}
            elif mode == "bicycling":
                # 骑行规划（v4，接口结构略有不同）
                url = f"{AMAP_BASE}/v4/direction/bicycling"
                params = {"key": AMAP_KEY, "origin": origin, "destination": destination}
            else:  # transit
                # 公交规划（v3，需要额外传城市参数）
                url = f"{AMAP_BASE}/v3/direction/transit/integrated"
                params = {"key": AMAP_KEY, "origin": origin, "destination": destination, "city": city}

            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            # 提取核心信息（不同接口返回结构略有差异）
            if mode == "bicycling":
                # 骑行接口用 errcode 而非 status 表示是否成功
                if data.get("errcode") != 0:
                    return {"ok": False, "error": data.get("errmsg", "骑行规划失败")}
                paths = data.get("data", {}).get("paths", [])
            elif mode == "transit":
                if data.get("status") != "1":
                    return {"ok": False, "error": data.get("info", "公交规划失败")}
                # 公交规划返回 transits（候选方案列表）
                paths = data.get("route", {}).get("transits", [])
            else:
                # 驾车/步行 统一用 status + paths
                if data.get("status") != "1":
                    return {"ok": False, "error": data.get("info", "规划失败")}
                paths = data.get("route", {}).get("paths", [])

            if not paths:
                return {"ok": False, "error": "无可用路线"}
            first = paths[0]  # 取第一条（最优）路线
            return {
                "ok": True,
                "mode": mode,
                "origin": origin,
                "destination": destination,
                "distance_m": int(first.get("distance", 0)),   # 总距离（米）
                "duration_s": int(first.get("duration", 0)),   # 预计时间（秒）
                "steps_summary": (first.get("steps") or [])[:3],  # 只返前 3 步摘要，节省 token
            }
    except httpx.RequestError as e:
        return {"ok": False, "error": f"高德 API 连接失败: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"路线规划失败: {e}"}


@mcp.tool()
async def nearby_search(
    location: str,
    keyword: str,
    radius: int = 1000,
    city: str = "北京",
) -> dict[str, Any]:
    """
    通用周边搜索（自定义关键词）。

    Args:
        location: 中心点，可以是地名"北京协和医院" 或 "lng,lat" 坐标
        keyword: 搜索关键词，如"咖啡"、"便利店"
        radius: 半径（米），默认 1000
        city: 城市（地名解析用），默认"北京"

    Returns:
        { "ok": bool, "count": int, "pois": [...] }
    """
    if err := _check_key():
        return err
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 判断 location 是否已是坐标（含逗号），否则地理编码
            if "," not in location:
                poi = await _geocode(client, location, city)
                if not poi:
                    return {"ok": False, "error": f"中心点未找到: {location}"}
                location = poi["location"]  # 转换为"lng,lat"坐标
            # 执行周边搜索
            pois = await _around(client, location, keyword, radius)
            return {"ok": True, "count": len(pois), "pois": [_fmt_poi(p) for p in pois]}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"高德 API 连接失败: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


async def _nearby_by_hospital(hospital: str, keyword: str, radius: int, city: str) -> dict[str, Any]:
    """
    内部组合函数：以医院为中心搜索周边指定关键词的 POI。
    供 nearby_hotels/restaurants/pharmacies/parking 四个工具复用。

    Args:
        hospital: 医院名称
        keyword: 搜索关键词（如"酒店|宾馆|旅馆"）
        radius: 搜索半径（米）
        city: 城市

    Returns:
        { "ok": bool, "hospital": {...}, "count": int, "pois": [...] }
    """
    if err := _check_key():
        return err
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 第一步：地理编码医院（可能命中缓存）
            hosp = await _geocode(client, hospital, city)
            if not hosp:
                return {"ok": False, "error": f"未找到医院: {hospital}"}
            # 第二步：以医院坐标为中心周边搜索
            pois = await _around(client, hosp["location"], keyword, radius)
            return {
                "ok": True,
                "hospital": _fmt_poi(hosp),  # 医院信息
                "count": len(pois),
                "pois": [_fmt_poi(p) for p in pois],  # 周边 POI 列表
            }
    except httpx.RequestError as e:
        return {"ok": False, "error": f"高德 API 连接失败: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"未知错误: {e}"}


@mcp.tool()
async def nearby_hotels(hospital: str, radius: int = 1000, city: str = "北京") -> dict[str, Any]:
    """医院附近酒店（住宿）查询。默认半径 1km。"""
    # 搜索关键词用 | 连接多个同义词，提高召回率
    return await _nearby_by_hospital(hospital, "酒店|宾馆|旅馆", radius, city)


@mcp.tool()
async def nearby_restaurants(hospital: str, radius: int = 500, city: str = "北京") -> dict[str, Any]:
    """医院附近餐饮查询。默认半径 500m。"""
    return await _nearby_by_hospital(hospital, "餐饮|餐厅|快餐", radius, city)


@mcp.tool()
async def nearby_pharmacies(hospital: str, radius: int = 1000, city: str = "北京") -> dict[str, Any]:
    """医院附近药店查询。默认半径 1km。"""
    return await _nearby_by_hospital(hospital, "药店|药房", radius, city)


@mcp.tool()
async def nearby_parking(hospital: str, radius: int = 500, city: str = "北京") -> dict[str, Any]:
    """医院附近停车场查询。默认半径 500m。"""
    return await _nearby_by_hospital(hospital, "停车场", radius, city)


if __name__ == "__main__":
    # 打印启动信息到 stderr（不干扰 MCP stdio 通信）
    print(f"[amap_mcp] Web API: {AMAP_BASE}, key配置: {'✓' if AMAP_KEY else '✗未配置'}", file=sys.stderr)
    # 以 stdio 传输方式启动 MCP Server（与 MCPClientPool 通过子进程 stdin/stdout 通信）
    mcp.run(transport="stdio")
