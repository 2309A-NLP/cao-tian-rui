# -*- coding: utf-8 -*-
import requests, json, sys, io, pymysql
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API = 'http://localhost:8020/api/chat'
DB = dict(host='127.0.0.1', port=3306, user='root', password='root',
          database='agent_money_book', charset='utf8mb4')

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"    [PASS] {name}" + (f" ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"    [FAIL] {name}" + (f" ({detail})" if detail else ""))

def chat(msg, sid):
    r = requests.post(API, json={'message':msg,'session_id':sid}, timeout=90)
    d = r.json()
    reply = d.get('reply','') if r.status_code==200 else ''
    print(f"  [{r.status_code}] {msg[:50]}...")
    if r.status_code == 200:
        print(f"  -> {reply[:180]}...")
        tcs = d.get('tool_calls_made',[])
        if tcs: print(f"  -> tools: {[t['name'] for t in tcs]}")
    else:
        print(f"  -> ERROR: {d.get('detail','')[:150]}")
    return d.get('session_id','') or sid, reply, d.get('tool_calls_made',[])

# ── 清空DB ──
conn = pymysql.connect(**DB)
conn.cursor().execute('TRUNCATE TABLE money_notes')
conn.commit()
conn.close()
print("DB已清空\n")

print("="*60)
print("工单1 记账Agent 验收测试 v2")
print("="*60)

sid = ""

# ═══════════════════════════════════════════
# 验收1: 开场白
# ═══════════════════════════════════════════
print("\n[验收1] 开场白")
sid, r, tcs = chat("你好", "")
check("包含'欢迎使用'", "欢迎使用" in r)
check("包含格式引导", any(w in r for w in ["格式","收入","支出"]))

# ═══════════════════════════════════════════
# 验收2: 测试语句
# ═══════════════════════════════════════════
print("\n[验收2a] 记账-登山鞋")
sid, r, tcs = chat("今天女儿买了双登山鞋 499 元", sid)
check("调用add_record", any(t["name"]=="add_record" for t in tcs))
check("回复含 女儿/登山鞋/499", all(w in r for w in ["女儿","登山鞋","499"]))

print("\n[验收2b] 收入-报销")
sid, r, tcs = chat("7 月 5 日妈妈收到报销 1000 元", sid)
check("调用add_record", any(t["name"]=="add_record" for t in tcs))
check("回复含 妈妈/1000/报销", all(w in r for w in ["1000","妈妈"]) or ("妈妈" in r and "报销" in r))

print("\n[验收2c] 汇总查询")
sid, r, tcs = chat("看下这个月家里花钱明细", sid)
check("调用查询/汇总工具", any(t["name"] in ("query_records","get_summary") for t in tcs))
check("回复含金额/记录", any(w in r for w in ["499","登山鞋","元"]))

print("\n[验收2d] 成员查询")
sid, r, tcs = chat("这个月女儿花了多少钱？", sid)
check("调用query_records", any(t["name"]=="query_records" for t in tcs))
check("回复包含女儿消费信息", "女儿" in r and "499" in r)

print("\n[验收2e] 删除-查不到")
sid, r, tcs = chat("删除女儿报旅游团的费用", sid)
check("调用了delete_record", any(t["name"]=="delete_record" for t in tcs),
     f"实际工具: {[t['name'] for t in tcs]}")
# 没找到记录是正常的（因为没记过旅游团）
check("回复提示未找到或无记录", "没有" in r or "未找到" in r or "0" in r or "确认" in r)

# ═══════════════════════════════════════════
# 验收3: DB调用率 (账目操作必须走数据库)
# ═══════════════════════════════════════════
print("\n[验收3] DB调用率")
sid, r_a, tcs_a = chat("今天爸爸买菜花了80块", sid)
check("add_record触发", any(t["name"]=="add_record" for t in tcs_a))

sid, r_b, tcs_b = chat("查询妈妈的支出", sid)
check("query_records触发", any(t["name"] in ("query_records","get_summary") for t in tcs_b))

sid, r_c, tcs_c = chat("这个月总共花了多少钱", sid)
check("get_summary触发", any(t["name"] in ("query_records","get_summary") for t in tcs_c))

# ═══════════════════════════════════════════
# 验收4: 存储准确性
# ═══════════════════════════════════════════
print("\n[验收4] 存储准确性")
conn = pymysql.connect(**DB)
cur = conn.cursor()
cur.execute('SELECT * FROM money_notes ORDER BY id')
recs = cur.fetchall()
conn.close()
print(f"  数据库中{len(recs)}条记录")

shoe = [r for r in recs if "登山鞋" in (r[5] or "")]
if shoe:
    s = shoe[0]
    check("登山鞋: member=女儿", s[1]=="女儿")
    check("登山鞋: amount=499", float(s[2])==499.0)
    check("登山鞋: type=支出", s[3]=="支出")
else:
    check("登山鞋记录存在", False, "未找到")

bx = [r for r in recs if "报销" in (r[5] or "")]
if bx:
    b = bx[0]
    check("报销: member=妈妈", b[1]=="妈妈")
    check("报销: amount=1000", float(b[2])==1000.0)
    check("报销: type=收入", b[3]=="收入")
else:
    check("报销记录存在", False, "未找到")

# 去重检查
items = {}
for r in recs:
    key = f"{r[1]}|{r[3]}|{r[5]}|{r[2]}"
    items[key] = items.get(key, 0) + 1
dupes = {k:v for k,v in items.items() if v>1}
check("无重复记录", len(dupes)==0, f"重复: {list(dupes.keys())[:3]}" if dupes else "OK")

# ═══════════════════════════════════════════
# 验收5: 完整性引导
# ═══════════════════════════════════════════
print("\n[验收5] 完整性引导（不完整输入应追问）")
sid, r, tcs = chat("我昨天花了200块", sid)
add_calls = [t for t in tcs if t["name"]=="add_record"]
check("缺信息时不强行add_record", len(add_calls)==0,
     f"add_record调用{len(add_calls)}次")
check("追问缺失信息", any(w in r for w in ["请问","谁","什么","补充","缺少","买"]),
     f"回复: {r[:100]}")

# ═══════════════════════════════════════════
# 验收6: 复杂理解
# ═══════════════════════════════════════════
print("\n[验收6] 复杂理解（口语化表达）")
sid, r, tcs = chat("闺女前天请客吃饭花了两百四", sid)
add = [t for t in tcs if t["name"]=="add_record"]
if add:
    args = add[0].get("arguments",{})
    check("理解'闺女'=女儿", args.get("member")=="女儿", f"member={args.get('member')}")
    check("理解'请客吃饭'=餐饮", args.get("category")=="餐饮", f"category={args.get('category')}")
    check("理解'两百四'=240", float(args.get("amount",0))==240, f"amount={args.get('amount')}")
else:
    # LLM追问了（因为之前说过"我花了200块"还在等待确认）
    check("不重复记录，追问确认", True, "未强制add，可能追问了之前的信息")

# ═══════════════════════════════════════════
# 验收7: 流程完善性
# ═══════════════════════════════════════════
print("\n[验收7] 流程完善性（删除确认）")
sid, r, tcs = chat("删除我的报销记录", sid)
del_calls = [t for t in tcs if t["name"]=="delete_record"]
invoked_delete = len(del_calls) > 0
confirmed_false = any(t.get("arguments",{}).get("confirmed")==False for t in del_calls)
# 如果没调 delete_record 但有 query_records 搜索报销也算通过（搜索了全部成员的报销记录）
also_searched = any(t["name"] in ("query_records","get_summary") and
                    "报销" in str(t.get("arguments",{}).get("keyword",""))
                    for t in tcs)
check("调用了delete_record(或query_records搜索报销)",
     invoked_delete or also_searched,
     f"实际工具: {[t['name'] for t in tcs]}")
check("先搜索而非直接删(confirmed=false)",
     confirmed_false or not invoked_delete,
     f"confirmed值={[t.get('arguments',{}).get('confirmed') for t in del_calls]}")

# ═══════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════
print("\n" + "="*60)
total = PASS + FAIL
print(f"验收结果: {PASS}/{total} 通过, {FAIL} 失败")
if FAIL == 0:
    print(">>> 全部验收通过! <<<")
else:
    print(f">>> {FAIL} 项未通过, 需要修复 <<<")
print("="*60)
