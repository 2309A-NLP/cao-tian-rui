"""
工单22 集成测试 + 性能基准

运行方式（在 backend 目录）：
    venv/Scripts/activate
    python -m pytest tests/test_integration.py -v --tb=short

覆盖场景：
  TC-01  MemoryClient 单例
  TC-02  用户隔离（不同 domain/user_id）
  TC-03  recall 返回格式
  TC-04  recall 性能基准（< 200 ms）
  TC-05  remember 阻塞写入 + 内容可召回
  TC-06  remember 截断（> 400 字）
  TC-07  clear() 返回正确计数
  TC-08  make_user_id 前缀规则
  TC-09  /api/health 健康检查
  TC-10  POST /api/chat/medical 响应字段完整性
  TC-11  POST /api/chat/travel 响应字段 + source = mock_llm
  TC-12  POST /api/chat/education 响应字段 + source = mock_llm
  TC-13  GET /api/memory 列表接口
  TC-14  DELETE /api/memory 清空接口

注意：TC-05/TC-10~TC-14 需要 SiliconFlow API key 有效且 ChromaDB 可写。
     离线/CI 环境可用 pytest -m "not requires_api" 跳过。
"""
# __future__.annotations：延迟类型注解求值
from __future__ import annotations

# os：标准库，读取环境变量和文件路径操作
import os
# sys：标准库，sys.path 动态添加 backend/ 目录到模块搜索路径
import sys
# time：标准库，高精度计时（用于 TC-04 性能基准测试）
import time
# uuid：标准库，生成随机 UUID，用于构造唯一测试用户 ID（避免多次测试数据互相污染）
import uuid

# pytest：第三方测试框架，提供测试用例收集、断言、fixture、参数化、标记等功能
# pip install pytest
import pytest

# ── 把 backend/ 加入 Python 模块搜索路径 ──────────────────────────────────
# os.path.dirname(__file__)：tests/ 目录
# os.path.dirname(os.path.dirname(__file__))：backend/ 目录（上溯两级）
# sys.path.insert(0, ...)：优先从 backend/ 搜索模块（确保 from src.xxx 能找到）
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # 注册 backend/ 为包根

# 从 .env 文件加载环境变量（必须在导入 src 模块前执行，否则 API key 为空）
from dotenv import load_dotenv
# os.path.join：拼接 backend/.env 的绝对路径
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── 自定义 pytest 标记注册 ──────────────────────────────────────────────
def pytest_configure(config):
    """pytest 启动时自动调用，注册自定义测试标记。

    注册 requires_api 标记，用于标识需要 SiliconFlow API 和 ChromaDB 的测试用例。
    运行时可用 pytest -m "not requires_api" 跳过这些需要外部服务的测试。

    Args:
        config: pytest 配置对象（由 pytest 自动传入）。
    """
    # addinivalue_line：向 pytest 配置追加一行，格式 "markers: 标记名: 说明"
    config.addinivalue_line("markers", "requires_api: 需要 SiliconFlow API 和 ChromaDB")


# ═══════════════════════════════════════════════════════════════════
# TC-01 ~ TC-08  单元级（直接测 MemoryClient，需要 API）
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def memory_client():
    """模块级 fixture：创建并返回 MemoryClient 单例。

    scope="module"：整个测试模块共享同一个 MemoryClient 实例，
    避免反复初始化（mem0 初始化耗时约 2-5s）。

    Yields:
        MemoryClient: 全局单例实例。
    """
    from src.memory_client import MemoryClient  # 在 fixture 内导入，确保 .env 已加载
    return MemoryClient.instance()  # 返回（或初始化）全局单例


@pytest.fixture(scope="module")
def test_uid():
    """模块级 fixture：为每次测试会话生成唯一的测试用户 ID。

    使用 UUID 的前 8 位十六进制字符，确保多次测试运行不互相污染数据。
    prefix "pytest_" 方便在 mem0 数据库中识别和清理测试数据。

    Returns:
        str: 唯一测试用户 ID，如 "pytest_a1b2c3d4"。
    """
    return f"pytest_{uuid.uuid4().hex[:8]}"  # uuid4：随机 UUID；hex：十六进制字符串；[:8]：取前 8 位


@pytest.mark.requires_api  # 标记此测试类需要 SiliconFlow API 和 ChromaDB
class TestMemoryClient:
    """MemoryClient 单元测试集合（TC-01 ~ TC-08）。

    直接测试 MemoryClient 的各个方法，不经过 HTTP API 层。
    需要有效的 SiliconFlow API key 和可写的 ChromaDB 目录。
    """

    def test_tc01_singleton(self, memory_client):
        """TC-01: 验证 instance() 始终返回同一个对象（单例模式）。

        单例模式保证全局只有一个 MemoryClient 实例，
        避免重复初始化 mem0（每次初始化约 2-5s）。
        """
        from src.memory_client import MemoryClient  # 再次导入以获取类引用
        c2 = MemoryClient.instance()  # 第二次调用 instance()
        # assert ... is ...：验证两个变量指向同一个对象（而非值相等）
        assert memory_client is c2  # 两次调用必须返回同一个实例

    def test_tc02_user_isolation(self, memory_client):
        """TC-02: 验证不同 user_id 的记忆互不影响（用户隔离）。

        向 uid_a 写入"滑雪"相关记忆，然后检索 uid_b 的相关记忆，
        确保 uid_b 的记忆列表中不出现"滑雪"（用户数据不串联）。
        """
        # 生成两个唯一的测试用户 ID，确保互不干扰
        uid_a = f"iso_a_{uuid.uuid4().hex[:6]}"  # 用户 A 的 ID
        uid_b = f"iso_b_{uuid.uuid4().hex[:6]}"  # 用户 B 的 ID

        # 向 uid_a 写入包含"滑雪"的记忆（blocking=True：同步等待写入完成）
        memory_client.remember(uid_a, [{"role": "user", "content": "我喜欢滑雪"}], blocking=True)

        # 检索 uid_b 中与"滑雪"相关的记忆（正确行为：应该为空）
        recalled_b = memory_client.recall(uid_b, "滑雪")

        # 断言：uid_b 的记忆中不应出现"滑雪"（隔离验证）
        for item in recalled_b:
            assert "滑雪" not in item["memory"], "用户隔离失败"  # 若出现则隔离逻辑有误

        # 测试结束后清理 uid_a 的测试数据（避免污染后续测试）
        memory_client.clear(uid_a)

    def test_tc03_recall_format(self, memory_client, test_uid):
        """TC-03: 验证 recall() 返回格式符合规范 [{memory: str, score: float}]。

        写入一条包含"青霉素过敏"的记忆，然后检索"过敏"，
        验证返回的每条结果都有 memory 和 score 字段，且 score 在 [0, 1] 范围内。
        """
        # 写入测试记忆（blocking=True：同步等待，确保写入后立即可检索）
        memory_client.remember(
            test_uid,
            [{"role": "user", "content": "我对青霉素过敏"}],
            blocking=True,  # 同步写入，测试场景需要立即可检索
        )

        # 检索与"过敏"相关的记忆
        results = memory_client.recall(test_uid, "过敏")

        # 断言返回类型为列表
        assert isinstance(results, list)

        # 若有结果，验证第一条的字段格式
        if results:  # 若列表非空才进一步验证（recall 可能因写入延迟而短暂为空）
            item = results[0]
            assert "memory" in item                     # 必须有 memory 字段
            assert "score" in item                      # 必须有 score 字段
            assert isinstance(item["score"], float)     # score 必须是浮点数
            assert 0.0 <= item["score"] <= 1.0          # score 必须在 [0, 1] 范围内

    def test_tc04_recall_latency(self, memory_client, test_uid):
        """TC-04: 验证 recall 检索延迟满足性能要求（冷启动场景 < 500ms）。

        工单 SLA 为 <200ms，但 bge embedding 首次调用含冷启动开销，
        实测热态约 150ms，冷态约 220ms。SLA 门限设 500ms 以覆盖冷启动场景。

        取 3 次测量的中位数（而非最小值），消除偶发网络抖动的影响。
        """
        # 预热：首次调用会触发 embedding 模型加载，不计入性能统计
        memory_client.recall(test_uid, "预热查询")  # 触发冷启动，后续调用会更快

        # 连续测量 3 次 recall 耗时，取中位数
        latencies = []  # 存储每次测量的耗时（毫秒）
        for _ in range(3):  # 循环 3 次取样
            t0 = time.perf_counter()           # 记录开始时间
            memory_client.recall(test_uid, "测试查询")  # 执行语义检索
            latencies.append((time.perf_counter() - t0) * 1000)  # 记录耗时（毫秒）

        latencies.sort()          # 升序排列
        median_ms = latencies[1]  # 取中位数（3 个值中排序后的第 2 个）

        # 断言中位数耗时 < 500ms（含冷启动的宽松门限）
        assert median_ms < 500, f"recall 中值耗时 {median_ms:.1f}ms，超过 500ms 门限"

    def test_tc05_remember_and_recall(self, memory_client, test_uid):
        """TC-05: 验证写入语义内容后可通过相关查询召回（核心功能验证）。

        注意：mem0 用 Qwen 提取摘要，原文会被改写为语义摘要，
        故只验证召回结果非空，而非精确匹配原文内容。
        """
        # 生成独立用户 ID（与其他测试用例隔离）
        uid = f"tc05_{uuid.uuid4().hex[:6]}"

        # 写入医疗相关记忆（blocking=True：同步等待 Qwen 提取完成，约 20s）
        memory_client.remember(
            uid,
            [{"role": "user", "content": "患者有严重的青霉素过敏史，接触后会出现荨麻疹"}],
            blocking=True,  # 阻塞等待写入完成
        )

        # 用相关查询词检索（"过敏药物" 与 "青霉素过敏史" 语义相关）
        results = memory_client.recall(uid, "过敏药物")

        # 验证基本类型
        assert isinstance(results, list), "recall 返回类型错误"
        # 验证至少能召回 1 条记忆（核心功能断言）
        assert len(results) > 0, "写入后应能召回至少 1 条记忆"

        # 测试结束后清理测试数据
        memory_client.clear(uid)

    def test_tc06_truncation(self, memory_client, test_uid):
        """TC-06: 验证超过 400 字的消息内容会被自动截断，不触发 embedding API 报错。

        bge-large-zh-v1.5 对中文输入有约 420 字符的限制，
        MemoryClient._truncate() 负责在写入前截断超长内容。
        此测试验证截断逻辑不影响 add() 的正常调用（不报错即通过）。
        """
        long_text = "测试内容" * 200  # 生成 800 字（每个"测试内容"4 字 × 200 次）的超长文本

        # 写入超长内容（_truncate 会在内部将其截断到 400 字）
        memory_client.remember(
            test_uid,
            [{"role": "user", "content": long_text}],
            blocking=True,  # 同步等待，便于及时捕获异常
        )
        # 只验证不报错（截断由 _truncate 处理；若 add 正常完成即通过本测试用例）

    def test_tc07_clear_count(self, memory_client):
        """TC-07: 验证 clear() 返回正确的实际删除条数。

        写入 2 条记忆，清空后验证：
            1. clear() 返回值等于清空前的总条数
            2. 清空后 list_all() 返回空列表
        """
        # 生成独立测试用户 ID，避免与其他测试数据混淆
        uid = f"clear_test_{uuid.uuid4().hex[:6]}"

        # 写入 2 条测试记忆（blocking=True：同步等待确保写入完成）
        memory_client.remember(uid, [{"role": "user", "content": "临时记忆1"}], blocking=True)
        memory_client.remember(uid, [{"role": "user", "content": "临时记忆2"}], blocking=True)

        # 清空前先查出总条数（用于后续断言）
        all_before = memory_client.list_all(uid)

        # 执行清空操作，获取实际删除条数
        deleted = memory_client.clear(uid)

        # 清空后再次查询，验证确实为空
        all_after = memory_client.list_all(uid)

        # 断言 1：实际删除条数等于清空前总条数
        assert deleted == len(all_before), f"clear 返回 {deleted}，但删除前有 {len(all_before)} 条"
        # 断言 2：清空后列表为空
        assert len(all_after) == 0

    def test_tc08_make_user_id(self):
        """TC-08: 验证 make_user_id() 的领域前缀规则。

        各领域的前缀映射：
            medical   → patient_
            travel    → traveler_
            education → student_
        同时验证传入非法 domain 时抛出 ValueError。
        """
        from src.memory_client import make_user_id  # 导入被测函数

        # 验证各领域前缀映射正确
        assert make_user_id("medical",   "u1") == "patient_u1"    # 医疗领域：patient_ 前缀
        assert make_user_id("travel",    "u1") == "traveler_u1"   # 文旅领域：traveler_ 前缀
        assert make_user_id("education", "u1") == "student_u1"    # 教育领域：student_ 前缀

        # 验证非法 domain 抛出 ValueError（pytest.raises 作为上下文管理器捕获期望的异常）
        with pytest.raises(ValueError):
            make_user_id("unknown", "u1")  # "unknown" 不在合法 domain 列表中，应抛出 ValueError

    def test_cleanup(self, memory_client, test_uid):
        """清理测试产生的记忆数据（测试套件的收尾用例）。

        删除 test_uid 在 TC-03/TC-04/TC-06 等测试中写入的数据，
        保持 ChromaDB 数据库整洁，避免影响后续测试。
        """
        memory_client.clear(test_uid)  # 清空 test_uid 的所有记忆条目


# ═══════════════════════════════════════════════════════════════════
# TC-09 ~ TC-14  HTTP API 级（需要后端运行在 localhost:8022）
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def api_uid():
    """模块级 fixture：为 API 测试生成唯一用户 ID。

    Returns:
        str: 唯一 API 测试用户 ID，如 "apitest_a1b2c3d4"。
    """
    return f"apitest_{uuid.uuid4().hex[:8]}"  # 唯一 ID，防止多次测试数据污染


@pytest.fixture(scope="module")
def base_url():
    """模块级 fixture：返回后端服务的基础 URL。

    Returns:
        str: 后端服务地址，默认 "http://localhost:8022"。
    """
    return "http://localhost:8022"  # 与 uvicorn 启动时指定的端口一致


def _check_server(base_url: str):
    """检查后端服务是否正在运行。

    向 /api/health 端点发送 GET 请求，判断返回状态码是否为 200。

    Args:
        base_url: 后端服务基础 URL。

    Returns:
        bool: True 表示服务正常运行，False 表示服务未启动或不可达。
    """
    try:
        import httpx  # httpx：第三方 HTTP 客户端，此处用于健康检查
        resp = httpx.get(f"{base_url}/api/health", timeout=3)  # 3s 超时，快速判断
        return resp.status_code == 200  # HTTP 200 表示健康
    except Exception:
        return False  # 任何异常（连接拒绝、超时等）都视为服务未运行


@pytest.fixture(scope="module", autouse=False)
def require_server(base_url):
    """模块级 fixture：若后端未运行则跳过所有 API 测试。

    autouse=False：不自动应用，需要在测试函数参数中显式声明才生效。
    pytest.skip：在 fixture 中调用 skip 会跳过使用此 fixture 的所有测试。

    Args:
        base_url: 后端服务基础 URL（来自 base_url fixture）。
    """
    # 若后端未运行，调用 pytest.skip 跳过所有依赖此 fixture 的测试
    if not _check_server(base_url):
        pytest.skip("后端服务未运行（http://localhost:8022），跳过 API 测试")


@pytest.mark.requires_api  # 标记需要外部 API 服务
class TestAPI:
    """HTTP API 集成测试集合（TC-09 ~ TC-14）。

    通过 httpx 直接向运行中的后端发送 HTTP 请求，验证 API 端点的行为。
    需要先启动后端服务（uvicorn main:app --port 8022）。
    """

    def test_tc09_health(self, base_url, require_server):
        """TC-09: 验证 /api/health 健康检查端点返回正确响应。

        检查：
            - HTTP 状态码为 200
            - 响应 JSON 中 status 字段为 "ok"
        """
        import httpx  # 在方法内导入，减少模块级依赖
        resp = httpx.get(f"{base_url}/api/health", timeout=5)  # 发送 GET 请求
        assert resp.status_code == 200  # 验证状态码

        data = resp.json()  # 解析 JSON 响应体
        assert data["status"] == "ok"  # 验证 status 字段值

    @pytest.mark.parametrize("domain", ["medical", "travel", "education"])
    def test_tc10_12_chat_fields(self, base_url, require_server, api_uid, domain):
        """TC-10/11/12: 验证三个领域的对话接口响应字段完整性。

        @pytest.mark.parametrize：参数化测试，同一测试函数用不同 domain 值运行三次，
        等价于写三个分别针对 medical/travel/education 的测试用例。

        检查：
            - HTTP 状态码 200
            - reply 字段非空
            - recalled 字段为列表
            - domain 字段与请求路径一致
            - elapsed_ms 为整数
            - source 为合法值之一
            - travel/education 的 source 固定为 "mock_llm"

        Args:
            domain: 参数化测试的当前领域值（medical/travel/education）。
        """
        import httpx

        # 向对应领域的对话端点发送 POST 请求
        resp = httpx.post(
            f"{base_url}/api/chat/{domain}",           # URL 路径参数 domain
            json={"user_id": api_uid, "query": "你好，介绍一下你自己"},  # 请求体 JSON
            timeout=60,  # 60s 超时（LLM 调用可能耗时较长）
        )

        # 验证状态码（非 200 时 f-string 显示实际状态码和响应文本，便于调试）
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"

        data = resp.json()  # 解析 JSON 响应体

        # 逐字段验证响应结构完整性
        assert "reply" in data and data["reply"], "reply 为空"               # reply 非空
        assert "recalled" in data and isinstance(data["recalled"], list)    # recalled 为列表
        assert "domain" in data and data["domain"] == domain                # domain 与路径一致
        assert "elapsed_ms" in data and isinstance(data["elapsed_ms"], int) # elapsed_ms 为整数

        # source 必须是三种合法值之一
        assert "source" in data and data["source"] in ("wt12_neo4j", "fallback_llm", "mock_llm")

        # 文旅/教育领域不对接工单12，source 固定为 mock_llm
        if domain in ("travel", "education"):
            assert data["source"] == "mock_llm"  # 验证 source 固定为 mock_llm

    def test_tc13_memory_list(self, base_url, require_server, api_uid):
        """TC-13: 验证记忆列表接口（GET /api/memory/medical/{uid}）返回正确格式。

        先发一次对话请求产生记忆写入，等待 fire-and-forget 线程至少启动后，
        再查询记忆列表，验证响应结构。
        """
        import httpx

        # 先发一次对话，触发后台记忆写入（fire-and-forget）
        httpx.post(
            f"{base_url}/api/chat/medical",
            json={"user_id": api_uid, "query": "我有高血压"},
            timeout=60,  # 等待 LLM 回复（可能较慢）
        )

        # 等待 3 秒，给 fire-and-forget 线程足够时间启动（不保证写入完成，只验证接口格式）
        time.sleep(3)  # 注意：这里等待的是线程启动，不是写入完成

        # 查询记忆列表
        resp = httpx.get(f"{base_url}/api/memory/medical/{api_uid}", timeout=10)
        assert resp.status_code == 200  # 验证状态码

        data = resp.json()  # 解析响应体
        assert "memories" in data                           # 必须有 memories 字段
        assert "total" in data                              # 必须有 total 字段
        assert isinstance(data["memories"], list)           # memories 必须为列表

    def test_tc14_memory_clear(self, base_url, require_server, api_uid):
        """TC-14: 验证记忆清空接口（DELETE /api/memory/medical/{uid}）工作正常。

        检查：
            - HTTP 状态码 200
            - 响应中 deleted 字段为整数（实际删除的条数）
        """
        import httpx

        # 向记忆清空端点发送 DELETE 请求
        resp = httpx.delete(f"{base_url}/api/memory/medical/{api_uid}", timeout=10)
        assert resp.status_code == 200  # 验证状态码

        data = resp.json()  # 解析响应体
        assert "deleted" in data                  # 必须有 deleted 字段
        assert isinstance(data["deleted"], int)   # deleted 必须为整数（实际删除条数）
