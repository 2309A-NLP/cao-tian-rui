"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要

配置模块 —— 统一读取 .env，暴露给其它模块。
所有模块应 `from src import config` 后使用 config.XXX，不得自行调用 os.getenv。
"""
import os  # 标准库：读取进程环境变量（结合 dotenv 使用）

from dotenv import load_dotenv  # python-dotenv：从 .env 文件加载环境变量到 os.environ

load_dotenv()  # 加载当前工作目录的 .env 文件（不存在时无副作用）


# ── ASR 引擎选择 ─────────────────────────────────────────────────
# 支持三种引擎：
#   tingwu：阿里云通义听悟（合规备选路径，暂不通电）
#   xfyun ：讯飞 IAT 实时语音转写（本工单主路径）
#   mock  ：纯模拟数据，不连外网（开发/演示/CI 用）
ASR_ENGINE: str = os.getenv("ASR_ENGINE", "xfyun").lower()

# ── 讯飞 IAT 实时语音转写鉴权参数 ────────────────────────────────────────
# 来源：讯飞开放平台 → 我的应用 → 实时语音转写
XF_APP_ID: str = os.getenv("XF_APP_ID", "")        # 讯飞应用 ID
XF_API_KEY: str = os.getenv("XF_API_KEY", "")      # 讯飞 API Key
XF_API_SECRET: str = os.getenv("XF_API_SECRET", "") # 讯飞 API Secret（用于 HMAC-SHA256 签名）

# ── 硅基流动 LLM（摘要生成 + 兜底翻译）─────────────────────────────
# 硅基流动（siliconflow.cn）提供 OpenAI 兼容 API，托管 Qwen 等开源大模型
SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")  # 访问令牌
SILICONFLOW_MODEL: str = os.getenv(
    "SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"  # 默认使用 Qwen 7B（省 token）
)

# ── 翻译配置 ────────────────────────────────────────────────────────
# 逗号分隔的目标语言列表，如 "en" 或 "en,ja"
TRANSLATE_LANGUAGES: str = os.getenv("TRANSLATE_LANGUAGES", "en")
# 翻译总开关（前端也可在 WebSocket 初始化时覆盖目标语言）
TRANSLATE_ENABLED: bool = os.getenv("TRANSLATE_ENABLED", "true").lower() == "true"

# ── 通义听悟鉴权参数（合规兜底路径，暂不使用）─────────────────────────────────
# 来源：阿里云 RAM 控制台 → AccessKey 管理
ALIYUN_AK_ID: str = os.getenv("ALIYUN_AK_ID", "")          # 阿里云 AccessKey ID
ALIYUN_AK_SECRET: str = os.getenv("ALIYUN_AK_SECRET", "")  # 阿里云 AccessKey Secret
TINGWU_APP_KEY: str = os.getenv("TINGWU_APP_KEY", "")       # 通义听悟应用 Key
# 通义听悟 API 域名（北京区）
TINGWU_ENDPOINT: str = os.getenv("TINGWU_ENDPOINT", "tingwu.cn-beijing.aliyuncs.com")
# 通义听悟 API 版本
TINGWU_API_VERSION: str = os.getenv("TINGWU_API_VERSION", "2023-09-30")

# 回调基地址（本地开发时用 ngrok 转发得到公网 URL，让通义听悟可回调）
CALLBACK_BASE_URL: str = os.getenv("CALLBACK_BASE_URL", "http://localhost:8015")

# ── 工单对接地址 ─────────────────────────────────────────────────
WT11_URL: str = os.getenv("WT11_URL", "http://localhost:8011")  # 工单11：挂号管理
WT12_URL: str = os.getenv("WT12_URL", "http://localhost:8012")  # 工单12：健康咨询

# ── 运行模式 ─────────────────────────────────────────────────────
# MOCK_MODE=true 等价于 ASR_ENGINE=mock，为了向后兼容保留此选项
MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"
if MOCK_MODE:
    ASR_ENGINE = "mock"  # MOCK_MODE 优先级高于 ASR_ENGINE 配置

# ── 服务端口 ─────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8015"))  # FastAPI 监听端口

# ── 回调安全（S1 安全需求）────────────────────────────────────────
# 在 .env 中设置非空值后，/api/callback 接口会校验 X-Callback-Secret 请求头
# 若为空字符串，则跳过校验（仅限本地开发，生产环境必须配置）
CALLBACK_SECRET: str = os.getenv("CALLBACK_SECRET", "")

# ── CORS 允许来源（S3 安全需求）───────────────────────────────────
# 逗号分隔的前端允许来源列表；留空则退化为 ["*"]（仅限纯本地开发）
# 生产环境必须收窄到真实前端域名
_raw_origins: str = os.getenv(
    "ALLOWED_ORIGINS",
    # 默认允许常见本地开发端口（前端 8080 / 后端同源 8000）
    "http://localhost:8080,http://localhost:8000,http://127.0.0.1:8080,http://127.0.0.1:8000",
)
# 解析为列表，过滤空字符串（防止 split 产生空元素）
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
