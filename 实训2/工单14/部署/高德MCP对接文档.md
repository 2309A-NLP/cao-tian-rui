# 高德地图 MCP 对接文档

> 工单编号：人工智能 NLP-Agent 数字人项目-医疗智能体-MCP
> 交付物 №2：工单原文明确要求"高德地图 MCP 对接文档"

---

## 一、对接概览

本项目对高德地图的对接采用 **业务组合封装** 方案：直接调用高德 Web 服务 REST API，
在 `mcp_servers/amap_mcp.py` 中把"地理编码 → 周边搜索/路线规划"等多步操作组合为
面向医疗场景的业务化 MCP tool。

### 为什么不直接对接高德官方 SSE MCP？

高德官方于 2025 年发布了 SSE 传输的 MCP Server（https://mcp.amap.com/sse ）。
我们**没有直接使用**，原因：

| 维度 | 高德官方 SSE MCP | 我们的 amap_mcp（Web API + stdio） |
|---|---|---|
| 传输 | SSE，需要稳定长连接 | stdio，本地子进程 |
| 网络依赖 | 强依赖 | 弱依赖（每次调用一次 REST） |
| 工具粒度 | 原子 API（POI 搜索/路线规划） | 业务组合（`nearby_hotels(hospital="XXX")` 一次完成"搜医院→搜周边"） |
| 认证 | key 拼在 URL 里 | key 存 `.env`，不出现在 URL |
| 稳定性 | 依赖高德服务 | 我们自己控制超时、重试、错误格式 |

**保留兼容路径**：如果未来想切回官方 SSE，`.env` 里的 `AMAP_MCP_URL` 已经预留，
`mcp_config.json` 里 `amap` 条目的 `transport` 改为 `"sse"` 即可。

---

## 二、密钥申请与配置

高德开放平台：<https://console.amap.com/dev/key/app>

需要申请 **3 个 Key**：

| 变量 | 用途 | 申请类型 |
|---|---|---|
| `AMAP_API_KEY` | 后端（`amap_mcp.py`）调用 Web 服务 API | Web 服务 |
| `AMAP_JS_KEY` | 前端（`map.html`）加载 JS API 渲染地图 | Web 端 (JS API) |
| `AMAP_JS_SECURITY_CODE` | JS API 的安全密钥，配对 `AMAP_JS_KEY` | 与 JS Key 同页面获取 |

**⚠️ 注意**：Web 服务 Key 与 JS API Key 不能互用，是两套独立 Key。

配置示例（`.env`）：

```bash
AMAP_API_KEY=6a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d
AMAP_JS_KEY=7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c
AMAP_JS_SECURITY_CODE=abc123def456ghi789
```

---

## 三、使用的高德 Web 服务 API

### 3.1 关键 POI 搜索（地点搜索）

```text
GET https://restapi.amap.com/v3/place/text
参数：key, keywords, city, offset, extensions
用途：medical → 用医院名换坐标
```

被 `_geocode()` 内部使用，对应 tool: `hospital_search`

### 3.2 周边搜索

```text
GET https://restapi.amap.com/v3/place/around
参数：key, location(lng,lat), keywords, radius, offset, sortrule
用途：以医院坐标为中心搜索指定关键词
```

被 `_around()` 内部使用，对应 tools: `nearby_search / nearby_hotels / nearby_restaurants / nearby_pharmacies / nearby_parking`

### 3.3 路径规划

| 模式 | Endpoint |
|---|---|
| 驾车 | `https://restapi.amap.com/v3/direction/driving` |
| 步行 | `https://restapi.amap.com/v3/direction/walking` |
| 骑行 | `https://restapi.amap.com/v4/direction/bicycling` |
| 公交 | `https://restapi.amap.com/v3/direction/transit/integrated` |

对应 tool: `route_planning(mode=driving|walking|bicycling|transit)`

### 3.4 JS API（前端）

```html
<script>
  window._AMapSecurityConfig = { securityJsCode: "..." };  // 必须在 AMap JS 前设置
</script>
<script src="https://webapi.amap.com/maps?v=2.0&key=xxx&plugin=AMap.Driving,AMap.Walking,AMap.InfoWindow">
</script>
```

配套插件：
- `AMap.PlaceSearch`：POI 搜索
- `AMap.Driving / Walking / Riding / Transfer`：路径规划渲染
- `AMap.InfoWindow`：气泡弹窗

---

## 四、MCP 工具映射

Agent 通过 `mcp_servers/amap_mcp.py` 使用高德能力。所有 tool 都返回 `{ok: bool, ...}` 统一格式。

| Tool | 参数 | 说明 |
|---|---|---|
| `hospital_search` | name, city | 医院搜索 |
| `route_planning` | origin, destination, mode, city | 路线规划（4 模式） |
| `nearby_search` | location, keyword, radius, city | 通用周边（自定义关键词） |
| `nearby_hotels` | hospital, radius=1000, city | 医院附近酒店 |
| `nearby_restaurants` | hospital, radius=500, city | 医院附近餐饮 |
| `nearby_pharmacies` | hospital, radius=1000, city | 医院附近药店 |
| `nearby_parking` | hospital, radius=500, city | 医院附近停车 |

组合逻辑（`_nearby_by_hospital`）：先用 `_geocode` 把医院名换成坐标，再调 `_around`。

---

## 五、路由 Agent 如何使用

`mcp_client/router_agent.py` 通过 stdio 长连接 `amap_mcp` 子进程，
LLM (Qwen2.5-72B) 收到用户 query 后，通过 function calling 自主选择 tool 并组合参数：

```text
用户: "北京协和医院附近的酒店"
   ↓
LLM 分析 → 决定调 nearby_hotels(hospital="北京协和医院")
   ↓
amap_mcp 内部：
   1. GET /v3/place/text?keywords=北京协和医院 → 得到坐标 116.42,39.91
   2. GET /v3/place/around?location=116.42,39.91&keywords=酒店 → 得到 10 家酒店
   ↓
返回 { ok: true, hospital: {...}, count: 10, pois: [...] }
   ↓
LLM 生成自然语言回复
```

---

## 六、前端 REST 快捷通道

除了走 Agent（`/chat`），前端 `map.html` 还可以直接调用 `/api/amap/*`：

```text
GET  /api/amap/js-config?             → 返回前端加载 JS 需要的 key / security code
GET  /api/amap/hospital?name=xxx      → 医院搜索
GET  /api/amap/route?origin=&dest=&mode=  → 路线规划
GET  /api/amap/nearby?hospital=&category= → 周边（category=hotels|restaurants|pharmacies|parking）
```

这些端点在 `mcp_client/api_server.py` 里实现，内部走 `pool.call("nearby_hotels", {...})`
—— **与 Agent 共享同一个 MCP 通道**，没有第二套逻辑。

---

## 七、常见错误与排查

| 错误 | 原因 | 处理 |
|---|---|---|
| `AMAP_API_KEY 未配置` | `.env` 里没填 | 到高德开放平台申请 Web 服务 Key |
| `INVALID_USER_KEY` | Key 类型不对 | Web 服务 Key ≠ JS API Key，别搞混 |
| `USER_DAILY_QUERY_OVER_LIMIT` | 达到免费配额 | 高德免费版每日 3 万次，可申请商用配额 |
| 前端地图空白 | `AMAP_JS_KEY` 或 `AMAP_JS_SECURITY_CODE` 未配 | 检查 `/api/amap/js-config` 返回内容 |
| 路线规划失败 | 起终点解析失败 | 用完整地名或直接传 `lng,lat` 坐标 |

---

## 八、相关链接

- 高德开放平台：<https://lbs.amap.com/>
- Web 服务 API 文档：<https://lbs.amap.com/api/webservice/summary/>
- JS API 文档：<https://lbs.amap.com/api/jsapi-v2/summary/>
- 官方 MCP Server 说明：<https://lbs.amap.com/api/mcp-server/summary>

---

*文档版本：v1.0 / 2026-07-05*
