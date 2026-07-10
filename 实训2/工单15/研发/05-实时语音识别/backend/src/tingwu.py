"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要

通义听悟 HTTP API 封装，包含：
- Alibaba Cloud ACS3-HMAC-SHA256 请求签名
- CreateTask  创建实时记录任务（步骤1：获取 WebSocket 推流地址）
- CreateRecord 结束录制并触发离线分析（步骤3）
- GetTaskInfo  查询任务状态及结果（步骤4：轮询用）
- get_nls_token NLS Token 获取（方案B：NLS 实时转写备选路径）

注意：本文件为兜底路径（合规备选），工单主路径为讯飞 IAT（xfyun.py）。
"""
import base64  # 标准库：base64 编解码，用于 HMAC 签名结果的 Base64 编码（POP签名）
import hashlib  # 标准库：哈希算法，用于 SHA256/SHA1 摘要计算
import hmac as hmac_mod  # 标准库：HMAC-SHA256/HMAC-SHA1 消息认证码（请求签名核心）
import json  # 标准库：JSON 序列化（构造 API 请求体）
import urllib.parse  # 标准库：URL 编码，用于 POP 签名的参数编码
import uuid  # 标准库：生成随机 UUID（SignatureNonce 防重放攻击）
from datetime import datetime, timezone  # 标准库：UTC 时间戳（签名必须用 UTC）
from typing import Optional  # 标准库：类型注解，Optional[X] = X 或 None

# httpx 包：异步 HTTP 客户端（同步请求会阻塞事件循环）
import httpx

# 从配置模块读取阿里云/通义听悟鉴权参数
from src.config import (
    ALIYUN_AK_ID,          # 阿里云 AccessKey ID
    ALIYUN_AK_SECRET,      # 阿里云 AccessKey Secret（用于签名，切勿泄露）
    CALLBACK_BASE_URL,     # 通义听悟回调地址
    MOCK_MODE,             # Mock 模式开关
    TINGWU_API_VERSION,    # 通义听悟 API 版本（如 "2023-09-30"）
    TINGWU_APP_KEY,        # 通义听悟应用 Key
    TINGWU_ENDPOINT,       # 通义听悟 API 域名（如 tingwu.cn-beijing.aliyuncs.com）
    TRANSLATE_LANGUAGES,   # 翻译目标语言列表（逗号分隔字符串）
)


# ---------------------------------------------------------------------------
# Aliyun ACS3-HMAC-SHA256 签名工具
# 参考：https://help.aliyun.com/zh/sdk/product-overview/request-signature-v3
# ACS3 是阿里云新版签名算法，比旧版 POP/RPC 更安全（覆盖请求体 Hash）
# ---------------------------------------------------------------------------

def _sha256_hex(data: str | bytes) -> str:
    """
    计算 SHA-256 摘要并返回小写十六进制字符串。
    用于 ACS3 签名中的 BodyHash 和 CanonicalRequest 摘要。

    Args:
        data: 待摘要的字符串或字节

    Returns:
        64 字符小写十六进制字符串
    """
    if isinstance(data, str):
        data = data.encode()  # 字符串先转 UTF-8 字节
    return hashlib.sha256(data).hexdigest()


def _build_auth_headers(
    method: str,  # HTTP 方法（GET/PUT/POST），全大写
    path: str,    # URL 路径（如 "/openapi/tingwu/v2/tasks"）
    query: dict,  # URL 查询参数 dict（如 {"type": "realtime"}）
    body: str,    # 请求体 JSON 字符串（GET 时为空字符串 ""）
    action: str,  # 通义听悟 API action（如 "CreateTask"）
) -> dict:
    """
    构造带 ACS3-HMAC-SHA256 签名的请求头。

    ACS3 签名流程（四步法）：
    1. CanonicalRequest = Method + URI + QueryString + CanonicalHeaders + SignedHeaders + BodyHash
       CanonicalHeaders 是参与签名的 header key:value\\n 列表（按 key 字母序排列）
    2. StringToSign = "ACS3-HMAC-SHA256\\n" + HEX(SHA256(CanonicalRequest))
    3. Signature    = HEX(HMAC-SHA256(AK_SECRET, StringToSign))
    4. Authorization = ACS3-HMAC-SHA256 Credential=<AK_ID>,SignedHeaders=...,Signature=...

    Args:
        method: HTTP 方法
        path: URL 路径
        query: URL 查询参数
        body: 请求体字符串
        action: 通义听悟 API action 名称

    Returns:
        包含完整签名的请求头 dict，可直接传给 httpx 请求
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")  # ISO 8601 UTC 时间戳（签名必须用 UTC）
    body_hash = _sha256_hex(body)  # 请求体的 SHA-256 十六进制摘要（ACS3 特有，防篡改）

    # 参与签名的 header 集合（x-acs-* 为阿里云私有前缀）
    sign_headers: dict[str, str] = {
        "host": TINGWU_ENDPOINT,          # 请求域名（防中间人替换域名）
        "x-acs-action": action,            # API action（防 action 替换攻击）
        "x-acs-version": TINGWU_API_VERSION,  # API 版本
        "x-acs-date": date_str,            # 时间戳（防重放攻击，有效窗口通常 15 分钟）
        "x-acs-content-sha256": body_hash, # 请求体摘要（防数据篡改）
    }
    sorted_keys = sorted(sign_headers.keys())  # 按字母序排列（ACS3 规范要求）
    # 规范化 header 字符串：每行格式 "key:value\n"（小写 key，无多余空格）
    canonical_headers = "".join(f"{k}:{sign_headers[k]}\n" for k in sorted_keys)
    # 签名 header 名列表（用分号连接，与 Authorization 中 SignedHeaders 一致）
    signed_headers_str = ";".join(sorted_keys)

    # 规范化查询字符串：按 key 字母序排列的 "k=v&k=v" 格式
    canonical_query = "&".join(f"{k}={v}" for k, v in sorted(query.items()))

    # 拼接规范化请求（6 段，换行符分隔）
    canonical_request = "\n".join([
        method.upper(),   # 1. HTTP 方法
        path,             # 2. URL 路径
        canonical_query,  # 3. 查询字符串
        canonical_headers, # 4. 规范化 header
        signed_headers_str, # 5. 签名 header 名列表
        body_hash,        # 6. 请求体摘要
    ])

    # 待签字符串：算法标识 + 规范化请求的 SHA-256 摘要
    string_to_sign = f"ACS3-HMAC-SHA256\n{_sha256_hex(canonical_request)}"

    # HMAC-SHA256 签名：用 AK_SECRET 对 string_to_sign 签名，结果转十六进制
    # hmac_mod.new 返回 HMAC 对象，.hexdigest() 输出十六进制字符串
    signature = hmac_mod.new(
        ALIYUN_AK_SECRET.encode(),  # AK_SECRET 作为 HMAC 密钥
        string_to_sign.encode(),    # 待签字符串
        hashlib.sha256,             # 哈希函数
    ).hexdigest()

    # 合并签名头和通用头（content-type + Authorization）
    headers = dict(sign_headers)
    headers["content-type"] = "application/json"
    # Authorization 头格式：ACS3-HMAC-SHA256 Credential=...,SignedHeaders=...,Signature=...
    headers["authorization"] = (
        f"ACS3-HMAC-SHA256 Credential={ALIYUN_AK_ID},"
        f"SignedHeaders={signed_headers_str},"
        f"Signature={signature}"
    )
    return headers


# ---------------------------------------------------------------------------
# 通义听悟 API 封装
# ---------------------------------------------------------------------------

async def create_realtime_task(
    task_key: str,                      # 业务层任务 key（自定义，用于幂等）
    callback_url: Optional[str] = None, # 可选回调 URL（异步通知结果）
    source_language: str = "cn",         # 音频语言（cn=普通话）
) -> dict:
    """
    创建实时记录任务（步骤 1）。
    调用通义听悟 PUT /openapi/tingwu/v2/tasks?type=realtime。

    Args:
        task_key: 业务自定义任务标识（用于幂等控制）
        callback_url: 任务完成后通义听悟主动回调的 URL（可选）
        source_language: 音频语言（cn=普通话，en=英语等）

    Returns:
        {"task_id": str, "meeting_join_url": str}
        meeting_join_url 是客户端推送 PCM 音频的 WebSocket 地址
    """
    if MOCK_MODE:
        # Mock 模式：不连接真实 API，返回伪造的 task_id 和 WSS 地址
        mock_id = str(uuid.uuid4())
        return {
            "task_id": mock_id,
            "meeting_join_url": f"wss://mock-tingwu/{mock_id}",
        }

    path = "/openapi/tingwu/v2/tasks"
    query = {"type": "realtime"}  # 查询参数：实时任务类型

    # 构造请求体：指定音频格式/语言/扬声器数量/翻译目标语言等
    body_data: dict = {
        "AppKey": TINGWU_APP_KEY,
        "Input": {
            "Format": "pcm",                 # 音频格式：原始 PCM（L16）
            "SampleRate": 16000,              # 采样率：16kHz
            "SourceLanguage": source_language,
            "TaskKey": task_key,
            "ProgressiveCallbacksEnabled": False,  # 不开启实时进度回调（减少流量）
        },
        "Parameters": {
            "Transcription": {
                "DiarizationEnabled": True,   # 开启说话人分离（医患对话需区分发言人）
                "SpeakerCount": 2,            # 假设 2 人对话（医生+患者）
            },
            "Translation": {
                "TargetLanguages": TRANSLATE_LANGUAGES,  # 翻译目标语言列表
            },
            "AutoChapters": {"ChaptersEnabled": True},  # 自动章节切分
            "Summarization": {"Types": ["Paragraph", "Conversational"]},  # 摘要类型
            "MeetingAssistance": {"Types": ["ActionItems", "KeyInformation"]},  # 待办+关键信息
        },
    }

    if callback_url:
        # 若提供回调 URL，配置通义听悟在任务完成后 POST 通知
        body_data["Parameters"]["CallbackConfig"] = {
            "CallbackUrl": callback_url,
            "CallbackSecret": "",  # 可配置回调签名（此处留空）
        }

    body = json.dumps(body_data, ensure_ascii=False)  # 序列化请求体
    headers = _build_auth_headers("PUT", path, query, body, "CreateTask")  # 生成 ACS3 签名头

    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"https://{TINGWU_ENDPOINT}{path}",
            params=query,        # URL 查询参数（?type=realtime）
            headers=headers,     # 包含 ACS3 签名的请求头
            content=body.encode(),  # 请求体字节（注意不能用 json= 参数，已手动序列化）
            timeout=30,
        )
        if resp.status_code != 200:
            raise Exception(f"Tingwu API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return {
            "task_id": data["Data"]["TaskId"],
            "meeting_join_url": data["Data"]["MeetingJoinUrl"],  # 客户端推音频的 WSS URL
        }


async def create_record(task_id: str) -> dict:
    """
    结束实时录制，触发离线分析（步骤 3）。
    调用 POST /openapi/tingwu/v2/tasks/{task_id}/records。
    关闭 WebSocket 音频推流后调用此接口，通义听悟开始生成章节/摘要等。

    Args:
        task_id: CreateTask 返回的任务 ID

    Returns:
        API 响应 JSON（通常 {"code": "0"}）
    """
    if MOCK_MODE:
        return {"code": "0", "message": "ok"}  # Mock 模式直接返回成功

    path = f"/openapi/tingwu/v2/tasks/{task_id}/records"
    body = json.dumps({"AppKey": TINGWU_APP_KEY})  # 请求体只需 AppKey
    headers = _build_auth_headers("POST", path, {}, body, "CreateRecord")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://{TINGWU_ENDPOINT}{path}",
            headers=headers,
            content=body.encode(),
            timeout=30,
        )
        resp.raise_for_status()  # 非 200 抛出 HTTPStatusError
        return resp.json()


async def get_task_info(task_id: str) -> dict:
    """
    轮询任务状态及分析结果（步骤 4）。
    调用 GET /openapi/tingwu/v2/tasks/{task_id}。
    status: Ongoing | COMPLETED | FAILED
    COMPLETED 时 result 包含转写、翻译、章节、摘要等。

    Args:
        task_id: CreateTask 返回的任务 ID

    Returns:
        {TaskId, TaskStatus, Result: {Transcription, Translation, Chapters, Summarization}}
    """
    if MOCK_MODE:
        # Mock 模式：返回完整的医患对话示例数据
        return {
            "TaskId": task_id,
            "TaskStatus": "COMPLETED",
            "Result": {
                "Transcription": {
                    "Paragraphs": [
                        {
                            "ParagraphId": 1,
                            "Speaker": "发言人1",
                            "BeginTime": 0,
                            "EndTime": 5200,
                            "Words": [
                                {"Text": "医生您好，我最近头疼发烧已经两天了。", "StartTime": 0, "EndTime": 3000},
                            ],
                        },
                        {
                            "ParagraphId": 2,
                            "Speaker": "发言人2",
                            "BeginTime": 5500,
                            "EndTime": 12000,
                            "Words": [
                                {"Text": "我看一下，体温多少？有没有咳嗽？", "StartTime": 5500, "EndTime": 8000},
                            ],
                        },
                    ]
                },
                "Translation": {
                    "Paragraphs": [
                        {"Speaker": "发言人1", "TranslatedText": "Hello doctor, I have had a headache and fever for two days."},
                        {"Speaker": "发言人2", "TranslatedText": "Let me check. What is your temperature? Do you have a cough?"},
                    ]
                },
                "Chapters": [
                    {
                        "ChapterId": 1,
                        "Headline": "就诊主诉",
                        "BeginTime": 0,
                        "Summary": "患者主诉头痛发烧持续两天，医生问诊体温及症状。",
                    }
                ],
                "Summarization": {
                    "Paragraph": "患者陈述头痛发烧两天，医生询问体温和伴随症状，初步判断为感冒，建议休息并开具退烧药。",
                    "ActionItems": ["服用布洛芬退烧", "多喝水休息", "三天后复诊"],
                    "KeyInformation": ["症状：头痛发烧", "持续时间：两天", "初步诊断：感冒"],
                },
            },
        }

    path = f"/openapi/tingwu/v2/tasks/{task_id}"
    # GET 请求无 body（传空字符串），query 参数也为空
    headers = _build_auth_headers("GET", path, {}, "", "GetTaskInfo")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://{TINGWU_ENDPOINT}{path}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("Data", {})  # 通义听悟响应包裹在 Data 字段中


# ---------------------------------------------------------------------------
# NLS（智能语音交互）Token 获取
# 方案B：用 NLS 实时转写替代通义听悟 WebSocket 音频推流
# NLS 签名使用旧版 POP/RPC 格式（HMAC-SHA1），与 ACS3 不同
# ---------------------------------------------------------------------------

def _pop_sign(method: str, params: dict, ak_secret: str) -> str:
    """
    阿里云 POP/RPC 签名（HMAC-SHA1），专用于 NLS Token 接口。
    与 ACS3-HMAC-SHA256 不同，POP 签名对所有参数编码后计算签名。

    POP 签名流程：
    1. 按字母序排列所有参数（含 method、SignatureNonce 等防重放参数）
    2. URL 编码后拼成 "k=v&k=v" 格式
    3. StringToSign = "GET&%2F&" + URL编码(参数字符串)
    4. HMAC-SHA1(ak_secret + "&", StringToSign) → Base64 → Signature

    Args:
        method: HTTP 方法（通常为 "GET"）
        params: 所有请求参数 dict（不含 Signature 自身）
        ak_secret: 阿里云 AccessKey Secret

    Returns:
        Base64 编码的签名字符串
    """
    sorted_params = sorted(params.items())  # 按参数名字母序排列
    encoded_params = "&".join(
        # URL 编码参数名和值（safe='' 对所有字符编码，包括 "/" 和 "."）
        f"{urllib.parse.quote(str(k), safe='')}={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted_params
    )
    # POP 签名待签字符串：method + URL编码("/") + URL编码(参数字符串)
    string_to_sign = (
        f"{method}&{urllib.parse.quote('/', safe='')}"
        f"&{urllib.parse.quote(encoded_params, safe='')}"
    )
    # HMAC-SHA1 密钥 = AK_SECRET + "&"（POP 规范）
    key = (ak_secret + "&").encode()
    # hmac_mod.new 创建 HMAC 对象，hashlib.sha1 指定哈希算法
    # .digest() 返回原始字节，base64.b64encode 编码为 Base64 字符串
    sig = base64.b64encode(
        hmac_mod.new(key, string_to_sign.encode(), hashlib.sha1).digest()
    ).decode()
    return sig


async def get_nls_token() -> str:
    """
    获取 NLS 实时语音识别 Token（有效期约 24 小时）。
    使用 POP/RPC 签名调用阿里云 nls-meta 接口。
    Token 用于后续 NLS WebSocket 连接鉴权（方案B备用路径）。

    Returns:
        NLS Token 字符串（有效期约 24 小时）

    Raises:
        Exception: API 调用失败时抛出（含状态码和响应体）
    """
    if MOCK_MODE:
        return "mock-nls-token"  # Mock 模式返回伪 Token

    # 构造 POP 签名必需的所有参数
    params: dict = {
        "AccessKeyId": ALIYUN_AK_ID,        # 阿里云 AK ID
        "Action": "CreateToken",             # NLS Token API 的 action
        "Format": "JSON",                    # 响应格式
        "RegionId": "cn-shanghai",           # NLS 服务部署区域
        "SignatureMethod": "HMAC-SHA1",      # 签名算法（POP 规范）
        "SignatureNonce": uuid.uuid4().hex,  # 随机 nonce（防重放攻击）
        "SignatureVersion": "1.0",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": "2019-02-28",             # NLS Token API 版本
    }
    params["Signature"] = _pop_sign("GET", params, ALIYUN_AK_SECRET)  # 附加签名

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://nls-meta.cn-shanghai.aliyuncs.com/",  # NLS Token API 端点
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            raise Exception(f"NLS Token 获取失败 {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["Token"]["Id"]  # Token 在响应的 Token.Id 字段
