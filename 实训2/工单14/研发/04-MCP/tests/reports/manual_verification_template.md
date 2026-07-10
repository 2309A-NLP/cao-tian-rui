# 工单14 · 手工验证记录模板

> 每次实机跑通/联调后，复制此模板到 `manual_verification_<YYYYMMDD>.md`，逐条填结果。
> 目的：留下"这一版真的能跑"的书面证据，避免只靠记忆。

---

## 元信息

- **执行人**：
- **日期**：
- **04-MCP 代码版本**（可用 `git log -1 --oneline` 或最新改动日期代替）：
- **环境**：Windows / PowerShell / Python 3.x
- **上游启动状态**：
  - [ ] 00 门户（:8000）
  - [ ] 01 挂号（:8011）
  - [ ] 02 健康咨询（:8012）
  - [ ] 04 MCP 网关（:8014）
  - [ ] 03 影像（:8013）—— 未开工，占位
  - [ ] 05 语音（:8015）—— 未开工，占位

---

## A. 冷启动验证（不依赖上游）

**命令**：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m mcp_client.router_agent "北京协和医院在哪里"
```

**期望**：
- [ ] 5 个 pool 全部 ✓
- [ ] LLM 选中 `hospital_search`
- [ ] Reply 包含真实的地址、坐标、电话
- [ ] turns ≤ 3

**实际输出**（贴日志）：
```text

```

---

## B. 20 用例路由精度评测

**命令**：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m tests.eval_routing
```

**期望**：精度 ≥ 80%（工单验收线）

| 项 | 期望 | 实际 |
|---|---|---|
| 命中/总数 | ≥ 16/20 | / |
| 精度 | ≥ 80% | % |
| PASS/FAIL | PASS | |
| 报告文件 | `routing_eval_<ts>.md/.json` | |

---

## C. 性能压测

**命令**：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m tests.eval_performance
```

**期望**：L1 工具层 p95 < 500ms

| 工具 | p95 | 判定 |
|---|---|---|
| amap.hospital_search | ms | |
| amap.nearby_hotels | ms | |
| amap.nearby_restaurants | ms | |
| amap.route_planning | ms | |
| imaging.analyze_image | ms | |

**L2 端到端**（参考数据，非验收指标）：
- 典型 query 端到端时延：__ ~ __ s

---

## D. 容错评测

**命令**：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m tests.eval_robustness
```

**期望**：≥ 80% 用例得到"优雅回复"

| 项 | 期望 | 实际 |
|---|---|---|
| 通过数/总数 | ≥ 8/10 | / |
| 通过率 | ≥ 80% | % |

**关键用例检查**：
- [ ] R06 提示词注入未泄漏 SYSTEM_PROMPT
- [ ] R07/R08 未开工场景返回"暂未开放"

---

## E. Agent 思考过程演示

**命令**：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m tests.demo_thinking
```

**期望**：6 个典型场景全流程轨迹落盘

**报告文件**：`tests/reports/agent_thinking_<ts>.md`

---

## F. 前端 04-MCP 独立页面交互（无需上游）

**准备**：新开一个窗口跑网关（保持运行）：
```powershell
cd "F:\kimi  project\医疗agent1\04-MCP"
.\.venv\Scripts\Activate.ps1
python -m uvicorn mcp_client.api_server:app --host 0.0.0.0 --port 8014
```

浏览器打开 <http://localhost:8014/>

| # | 操作 | 期望 | 实际 |
|---|---|---|---|
| F1 | 加载首屏 | 双面板显示，顶栏有网关/高德两个绿灯 | |
| F2 | 搜索框输入`北京协和医院` → 搜索 | 医院卡出现（地址/电话/坐标）+ 地图定位红色 pin + 侧边默认加载"酒店"列表 | |
| F3 | 点右侧"餐饮"分类 tab | 侧边换餐厅列表，地图 pin 换色/换位 | |
| F4 | 点"药店" / "停车" | 同上 | |
| F5 | 点下方"驾车"路线 tab | 从"西直门"到医院的驾车路线画在地图上 | |
| F6 | 点医院卡"开始导航" | 新开高德导航页 uri.amap.com | |
| F7 | 搜索框输入`协和医院附近哪里能吃饭？`(长句问句) | AI 回答条弹出，地图自动切到餐饮分类 | |

---

## G. 从 00 门户跳转到 04 出行服务（跨工单联通）

**准备**：先启动 04 网关（同 F 步骤），另开一个窗口跑门户：
```powershell
cd "F:\kimi  project\医疗agent1\00-总界面框架"
python -m http.server 8000
```

浏览器打开 <http://localhost:8000/>，点左侧导航"出行服务"。

| # | 操作 | 期望 | 实际 |
|---|---|---|---|
| G1 | 从门户点击"出行服务" | iframe 加载出 8014 的地图页 | |
| G2 | 04 网关未启动时点击"出行服务" | offline 提示"请先启动 8014"+ 重新加载按钮 | |
| G3 | 在门户 iframe 里搜医院 | 与 F2 效果相同 | |
| G4 | 门户左侧切回"挂号管理" / "健康咨询" | 正常切换，无残留 | |

---

## H. 端到端联调（需 01 或 02 启动）

**准备**：另开窗口起 02（健康咨询）：
```powershell
cd "F:\kimi  project\医疗agent1\02-健康咨询"
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app:app --host 0.0.0.0 --port 8012
```

保持 8014 运行。在浏览器 <http://localhost:8014/> 输入`百日咳有什么症状`。

| # | 期望 | 实际 |
|---|---|---|
| H1 | AI 回答条给出百日咳的具体症状（来自 02 图谱） | |
| H2 | 04 网关日志出现 `knowledge_mcp` 调用 | |
| H3 | 02 服务日志出现 `/chat POST` 请求 | |

同理挂号（需 01 启动）：
```powershell
cd "F:\kimi  project\医疗agent1\01-挂号管理\backend"
.\venv\Scripts\Activate.ps1
python -m uvicorn src.app:app --host 0.0.0.0 --port 8011
```

浏览器输入`帮我查一下明天内科的号源` → 期望走 registration_mcp → 01 → 返回真号源列表。

---

## I. 总结

- 全部通过：[ ] 是 / [ ] 否
- 未通过项目：
- 遗留问题：
- 下一步动作：

---

*模板版本：v1.0 / 2026-07-06*
