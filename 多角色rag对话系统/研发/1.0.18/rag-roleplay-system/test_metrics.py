# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
RAG 系统指标测试脚本（性能、功能验证）
运行: python test_metrics.py

⚠️ 常改动的地方：
1. BASE_URL（服务地址与端口，默认 127.0.0.1:8001）
2. 测试问题列表（questions）可根据业务场景扩展
3. 测试用的用户 ID 和用户名（当前硬编码 user_id=1, username="test"）
4. 角色类型（测试聊天时默认为 "lawyer"/"friend"）
5. 上传测试文档的内容及格式

⚠️ 注意事项：
1. 运行前请确保服务已启动（python main.py）
2. 本脚本依赖 requests 和 sseclient-py 库（需安装：pip install requests sseclient-py）
3. 长期记忆测试依赖 Milvus 和 MySQL，需确保数据库已初始化
4. 上传文档测试会创建临时文件并在结束后删除，不影响现有知识库
5. 流式测试使用 SSE (Server-Sent Events)，需要服务端支持流式响应
"""

import requests
import time
import json

# ⚠️ 常改动：如果服务运行在其他地址或端口，修改此处
BASE_URL = "http://127.0.0.1:8001"


def test_health():
    """健康检查接口测试"""
    resp = requests.get(f"{BASE_URL}/api/health")
    print(f"健康检查: {resp.status_code} - {resp.json()}")
    return resp.status_code == 200


def test_chat_metrics():
    """普通聊天接口的性能指标测试（响应时间、答案长度、来源数量）"""
    print("\n" + "=" * 60)
    print("聊天指标测试")
    print("=" * 60)

    # ⚠️ 常改动：可增删或修改测试问题
    questions = [
        "捡到别人的失物怎么办？",
        "离婚需要什么条件？",
    ]

    for q in questions:
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={"user_id": 1, "username": "test", "message": q, "role": "lawyer"}
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            answer_len = len(data.get("answer", ""))
            sources_count = len(data.get("sources", []))

            print(f"\n问题: {q}")
            print(f"  响应时间: {elapsed:.0f}ms")
            print(f"  回答长度: {answer_len} 字符")
            print(f"  来源数量: {sources_count}")
            if sources_count > 0:
                print(f"  来源文件: {data['sources'][0].get('file', 'N/A')}")
        else:
            print(f"  错误: {resp.status_code}")


def test_upload_metrics():
    """上传文档接口的性能测试（耗时）"""
    print("\n" + "=" * 60)
    print("上传文档指标测试")
    print("=" * 60)

    # 创建临时测试文件
    with open("test_upload.txt", "w", encoding="utf-8") as f:
        f.write("""测试文档
这是一个用于测试知识库动态更新的文档。
内容包括：测试上传功能的各项指标。
""")

    start = time.time()
    with open("test_upload.txt", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/knowledge/upload",
            files={"file": ("test_upload.txt", f, "text/plain")},
            data={"auto_process": "true"}
        )
    elapsed = (time.time() - start) * 1000

    print(f"上传耗时: {elapsed:.0f}ms")
    print(f"响应: {resp.json()}")

    # 清理临时文件
    import os
    os.remove("test_upload.txt")


def test_conversation_metrics():
    """长期记忆功能测试：先告知身份，再询问，验证是否记住"""
    print("\n" + "=" * 60)
    print("记忆对话指标测试")
    print("=" * 60)

    # 第一轮：告诉身份
    resp1 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"user_id": 1, "username": "test", "message": "我是个老师", "role": "lawyer"}
    )
    print(f"第一轮（告知身份）: {resp1.json().get('answer', '')[:50]}...")

    # 第二轮：询问身份
    resp2 = requests.post(
        f"{BASE_URL}/api/chat",
        json={"user_id": 1, "username": "test", "message": "我是什么职业", "role": "lawyer"}
    )
    answer = resp2.json().get('answer', '')
    print(f"第二轮（询问职业）: {answer[:100]}...")

    # 判断是否记住了身份
    if "老师" in answer:
        print("✅ 长期记忆正常工作！")
    else:
        print("❌ 长期记忆可能未生效")


def test_stream_metrics():
    """流式接口指标测试：首字延迟、总耗时、数据块数、内容长度"""
    print("\n" + "=" * 60)
    print("流式输出指标测试")
    print("=" * 60)

    try:
        import sseclient
    except ImportError:
        print("❌ sseclient-py 未安装，请运行: pip install sseclient-py")
        return

    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json={"user_id": 1, "username": "test", "message": "你好", "role": "friend"},
        stream=True
    )

    chunk_count = 0
    first_chunk_time = None
    full_content = ""

    client = sseclient.SSEClient(resp)
    for event in client.events():
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
        chunk_count += 1
        try:
            data = json.loads(event.data)
            if data.get('content'):
                full_content += data['content']
            if data.get('done'):
                break
        except:
            pass

    total_time = (time.time() - start) * 1000

    print(f"首字延迟: {first_chunk_time * 1000:.0f}ms" if first_chunk_time else "首字延迟: N/A")
    print(f"总耗时: {total_time:.0f}ms")
    print(f"数据块数: {chunk_count}")
    print(f"总内容长度: {len(full_content)} 字符")


def test_rag_stats():
    """获取 RAG 统计信息接口"""
    print("\n" + "=" * 60)
    print("RAG 统计信息")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/api/rag/stats")
    print(f"统计: {resp.json()}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG 系统指标测试")
    print("=" * 60)

    if not test_health():
        print("服务未启动，请先运行 python main.py")
        exit(1)

    test_chat_metrics()
    test_upload_metrics()
    test_conversation_metrics()
    test_stream_metrics()
    test_rag_stats()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)