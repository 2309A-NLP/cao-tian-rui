"""
工单11 医疗挂号Agent - 全局配置 (MySQL 版)
所有配置均从 .env 文件读取，不硬编码敏感信息到代码中。
"""
import os           # 标准库：读取环境变量 os.getenv()
from pathlib import Path   # 标准库：跨平台路径操作
# python-dotenv：读取 .env 文件中的 KEY=VALUE 配置到 os.environ
from dotenv import load_dotenv

# 从当前文件（src/config.py）向上两级找到项目根目录的 .env 文件并加载
# resolve() 将相对路径转成绝对路径，parent.parent 向上走两层目录
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- LLM 配置 ---
# 百炼（主模型，优先级1）
BAILIAN_API_KEY  = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
BAILIAN_MODEL    = os.getenv("BAILIAN_MODEL", "qwen3.7-plus")

# 硅基流动（兜底，优先级2/3）
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")            # API 密钥（必须在 .env 中配置）
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")  # API 基础 URL
SILICONFLOW_MODEL    = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")         # 使用的主模型 ID

# --- MySQL 连接配置 ---
DB_HOST     = os.getenv("DB_HOST", "127.0.0.1")   # 数据库主机地址，默认本机
DB_PORT     = int(os.getenv("DB_PORT", "3306"))    # 端口，必须转为 int（getenv 返回字符串）
DB_USER     = os.getenv("DB_USER", "root")          # 数据库用户名
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")      # 数据库密码
DB_NAME     = os.getenv("DB_NAME", "medical_agent") # 使用的数据库名

# --- 业务常量 ---
# 系统支持的科室白名单，用于校验用户输入和 LLM 参数
VALID_DEPARTMENTS = [
    "内科", "外科", "儿科", "皮肤科", "眼科", "牙科", "消化内科", "心内科",
]
# 号源类型白名单（专家号/普通号）
VALID_TITLES     = ["专家", "普通"]
# 排班时间段白名单（对应数据库 schedule.sch_time_slot 字段值）
VALID_TIME_SLOTS = ["09:00-11:00", "14:00-17:00", "19:00-21:00"]

# 挂号费用映射（按号源类型定价），单位：元
REGISTER_FEE   = {"专家": 50, "普通": 15}
# 最大可预约天数（从今天算起，超出此范围的排班不允许挂号）
MAX_FUTURE_DAYS = 14
