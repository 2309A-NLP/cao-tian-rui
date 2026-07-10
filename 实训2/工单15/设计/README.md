# 工单15 · 医疗智能体实时语音识别、翻译与会议纪要

> 工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要
> 端口：**8015** · 状态：**MOCK 全链路 OK**，配 3 个环境变量即可切真实讯飞

---

## 目录

```
05-实时语音识别/
├── prototype.html              原型稿（保留）
├── 需求分析报告.md              已按PDF微调
├── 页面结构规范.md              前端严格对照
├── docs/
│   ├── 一页纸设计.md           架构决策记录
│   ├── 技术总结.md             WS vs HTTP、异步、回调、多线程 → 协程
│   └── 通义听悟对接说明.md      合规兜底 + 权衡评估 + 一键回切
├── backend/                    后端（FastAPI + venv）
│   ├── src/                    config / app / xfyun / tingwu
│   ├── tests/                  pytest 16/16 通过
│   ├── venv/                   本地隔离环境（.gitignore）
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                    默认 MOCK 模式
└── frontend/                   前端（纯 HTML/CSS/JS + AudioWorklet）
    ├── index.html              独立可跑
    ├── voice.css               design-tokens 提取的 CSS 变量层
    ├── voice.js                WS 客户端 + 事件渲染
    └── pcm-worklet.js          AudioWorklet 抽 16kHz PCM 40ms 帧
```

---

## 快速启动（MOCK 模式，无需任何凭据）

```powershell
cd "F:\kimi  project\医疗agent1\05-实时语音识别\backend"
.\venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8015
```

浏览器打开 http://127.0.0.1:8015/

- 点击麦克风按钮 → 授予麦克风权限
- 立即看到 4 句模拟医患对话（含翻译）
- Stop 后自动切"会议纪要"Tab，显示章节/待办/关键词

---

## 切到真实讯飞 IAT（3 步）

1. **申请凭据**：到 https://www.xfyun.cn 注册 → 创建应用 → 开通"实时语音转写"，拿 `APP_ID / API_KEY / API_SECRET`
2. **改 `.env`**：
   ```ini
   ASR_ENGINE=xfyun
   MOCK_MODE=false
   XF_APP_ID=<你的>
   XF_API_KEY=<你的>
   XF_API_SECRET=<你的>
   SILICONFLOW_API_KEY=<硅基流动 sk-...>    # 用于翻译+摘要
   TRANSLATE_ENABLED=true
   TRANSLATE_LANGUAGES=en
   ```
3. **重启服务**，`/health` 应返回 `"asr":"xfyun"`；对着麦克风说话，实时字幕会带真实识别文本 + LLM 翻译

---

## 切到通义听悟（合规验收路径）

参见 [`docs/通义听悟对接说明.md`](docs/通义听悟对接说明.md) §3.4 详细步骤。

---

## API 契约

### WebSocket `/ws/stream`

**客户端 → 服务端**

| 帧 | 内容 |
|---|---|
| text | `{"type":"init","lang":"zh_cn"}` 首帧 |
| binary | PCM 16kHz 16bit mono 每帧 40ms（1280 字节） |
| text | `{"type":"stop"}` 结束 |

**服务端 → 客户端**

| 事件 | 用途 |
|---|---|
| `started` | 会话建立 |
| `transcription` | 中间识别（增量） |
| `sentence_end` | 句末 |
| `translation` | 句级翻译 |
| `completed` | 全部完成 |
| `log` / `error` | 诊断 |

### REST

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 引擎/模式状态 |
| POST | `/api/session/start` | 创建会话（Agent 时序对齐） |
| POST | `/api/session/{id}/stop` | 停止会话 |
| GET | `/api/session/{id}/poll` | 轮询结果（通义听悟风格字段） |
| GET | `/api/session/{id}/summary` | 会议纪要视图 |
| POST | `/api/session/{id}/consult` | 转写发工单12健康咨询 |
| POST | `/api/callback` | 通义听悟风格回调接收端 |

---

## 测试

```powershell
cd backend
$env:MOCK_MODE = "true"
$env:ASR_ENGINE = "mock"
.\venv\Scripts\python.exe -m pytest tests\ -v
```

预期：**16 passed**（health / start / stop / callback×3 / poll×2 / summary×2 / consult×2 / translate×3）

---

## 与其他工单的关系

| 工单 | 关系 |
|---|---|
| 工单11 挂号（:8011） | 无直接依赖 |
| 工单12 健康咨询（:8012） | ✅ `/consult` 端点转发本工单转写全文 |
| 工单13 影像（:8013） | 无直接依赖 |
| 工单14 MCP 地图（:8014） | 无直接依赖 |
| 总界面（00-总界面框架） | 阶段一 iframe 嵌入；阶段二组件化，待五工单齐后统一 |

---

## Docker 打包

**本阶段不打包**。按项目约定：五个工单全部通过测试后一次性构建镜像，避免频繁改动重复更新容器。相关 Dockerfile 参考旧项目 `E:\16---实训2\15--工单15\Dockerfile`（已验证过）。

---

## 关键设计决策速查

参见 [`docs/一页纸设计.md`](docs/一页纸设计.md)：

- ASR 三态引擎切换 `ASR_ENGINE=tingwu|xfyun|mock`
- 前端无构建（纯 HTML/CSS/JS），CSS 变量层提取自 [`00-总界面框架/design-tokens.json`](../00-总界面框架/design-tokens.json)
- 音频管道：`AudioContext@16000Hz → MediaStreamSource → AudioWorkletNode(pcm-framer) → WebSocket binary`
- 并发模型：asyncio 协程（`gather` 上下行 + `create_task` 后台翻译/咨询）
- 回调 + 轮询两条结果获取路径都实现
