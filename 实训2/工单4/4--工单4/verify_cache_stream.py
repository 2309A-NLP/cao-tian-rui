"""
verify_cache_stream.py — 验证缓存、流式、异步接口是否正确

测试内容：
  [1] 缓存读写（无需API，mock query结果）
  [2] query_stream() 事件序列合法性（mock）
  [3] 招股书预分类仍然命中缓存（无API）
  [4] 异步接口可调用（asyncio.run）

运行方式：
  cd E:/16---实训2/4--工单4
  venv/Scripts/python verify_cache_stream.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── 测试1：缓存读写 ────────────────────────────────────────────
print("\n[1] 缓存读写测试")

import fund_agent as fa

# 直接操作缓存
fa._cache_set("测试问题A", "测试答案A")
assert fa._cache_get("测试问题A") == "测试答案A", "FAIL: cache set/get"
assert fa._cache_get("不存在的问题") is None, "FAIL: cache miss should be None"

# 缓存上限：写入 _CACHE_MAX+1 条，第一条应被淘汰
for i in range(fa._CACHE_MAX + 1):
    fa._cache_set(f"q{i}", f"a{i}")
assert fa._cache_get("q0") is None, "FAIL: oldest entry should be evicted"
assert fa._cache_get(f"q{fa._CACHE_MAX}") == f"a{fa._CACHE_MAX}", "FAIL: newest entry missing"

# 清空缓存（避免影响后续测试）
fa._answer_cache.clear()
print("  PASS")

# ── 测试2：query_stream() 事件序列（Mock FundAgent.query）────────
print("\n[2] query_stream 事件序列测试（mock）")

# Mock：让 query_stream 命中缓存路径（先写缓存再调stream）
fa._cache_set("已知问题", "已知答案")
events = list(fa.get_agent().query_stream("已知问题"))
assert len(events) == 1, f"FAIL: 缓存命中应只有1个事件，实际 {len(events)}"
assert events[0] == ("answer", "已知答案"), f"FAIL: 事件内容错误 {events[0]}"

# 招股书预分类路径（不需要API）
fa._answer_cache.clear()
events2 = list(fa.get_agent().query_stream("云南沃森生物竞争优势是什么"))
assert any(e[0] == "answer" and e[1] == "__PROSPECTUS__" for e in events2), \
    f"FAIL: 招股书问题应返回 __PROSPECTUS__，实际 {events2}"

# 验证招股书结果也被缓存了
assert fa._cache_get("云南沃森生物竞争优势是什么") == "__PROSPECTUS__", \
    "FAIL: 招股书结果应写入缓存"

fa._answer_cache.clear()
print("  PASS")

# ── 测试3：预分类缓存写入后 query() 命中缓存 ─────────────────
print("\n[3] 预分类缓存 → query() 命中缓存测试")

fa._cache_set("上海华铭智能首发战略配售", "__PROSPECTUS__")
result = fa.get_agent().query("上海华铭智能首发战略配售")
assert result == "__PROSPECTUS__", f"FAIL: 应返回缓存值 __PROSPECTUS__，实际 {result!r}"
fa._answer_cache.clear()
print("  PASS")

# ── 测试4：异步接口可调用 ─────────────────────────────────────
print("\n[4] 异步接口 query_async() 测试（缓存命中，无API调用）")

import asyncio

fa._cache_set("异步测试问题", "异步测试答案")

async def _test_async():
    result = await fa.query_async("异步测试问题")
    assert result == "异步测试答案", f"FAIL: async 结果错误 {result!r}"
    # 也测试实例方法
    result2 = await fa.get_agent().query_async("异步测试问题")
    assert result2 == "异步测试答案", f"FAIL: 实例 async 结果错误 {result2!r}"

asyncio.run(_test_async())
fa._answer_cache.clear()
print("  PASS")

# ── 测试5：cache eviction 不破坏其他接口 ────────────────────────
print("\n[5] 缓存淘汰后接口正常")

# 写入500条后再写1条，不应崩溃
for i in range(fa._CACHE_MAX):
    fa._cache_set(f"压测q{i}", f"压测a{i}")
fa._cache_set("最后一条", "最后答案")
assert fa._cache_get("最后一条") == "最后答案", "FAIL: eviction后最新条目缺失"
fa._answer_cache.clear()
print("  PASS")

print("\n==================================================")
print("全部验证通过！")
print("==================================================")
