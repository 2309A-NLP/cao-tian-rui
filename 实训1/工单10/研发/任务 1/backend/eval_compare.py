"""
eval_compare.py — 跨运行评估对比

读取 output/eval_history/ 下的两次评估结果，输出指标 delta（涨/跌），
用于量化"这次改动到底让检索/延迟变好还是变坏"。

用法：
  python eval_compare.py                 # 对比最近两次运行
  python eval_compare.py A.json B.json    # 对比指定两个结果文件（B 相对 A 的变化）
  python eval_compare.py --list           # 列出所有历史运行
"""
import os
import sys
import json
import glob


def _history_dir() -> str:
    """定位 output/eval_history（先看 CWD，再看脚本上级目录）"""
    candidates = [
        os.path.join(os.getcwd(), "output", "eval_history"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "eval_history"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "eval_history"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return candidates[0]


def _list_runs() -> list[str]:
    """按文件名（含时间戳）排序的历史结果文件"""
    return sorted(glob.glob(os.path.join(_history_dir(), "eval_*.json")))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_delta(old: float, new: float, pct: bool = False, lower_better: bool = False) -> str:
    """格式化单个指标的 旧→新 (Δ)，带涨跌箭头"""
    d = new - old
    arrow = "→"
    if abs(d) > 1e-9:
        improved = (d < 0) if lower_better else (d > 0)
        arrow = "↑" if d > 0 else "↓"
        mark = "✅" if improved else "⚠️"
    else:
        mark = "  "
    if pct:
        return f"{old*100:5.1f}% {arrow} {new*100:5.1f}%  ({d*100:+.1f}pt) {mark}"
    return f"{old:7.1f} {arrow} {new:7.1f}  ({d:+.1f}) {mark}"


def compare(path_a: str, path_b: str):
    a, b = _load(path_a), _load(path_b)
    print("=" * 78)
    print("  评估结果对比 (A → B)")
    print("=" * 78)
    print(f"  A: {os.path.basename(path_a)}  [{a.get('timestamp','?')}] mode={a.get('retrieval_mode','?')}")
    print(f"  B: {os.path.basename(path_b)}  [{b.get('timestamp','?')}] mode={b.get('retrieval_mode','?')}")
    print()

    # ── 答案通过率 ──
    print("  答案质量:")
    print(f"    通过率        {_fmt_delta(a.get('answer_pass_rate',0), b.get('answer_pass_rate',0), pct=True)}")

    # ── 检索质量 ──
    print("\n  检索质量 (越高越好):")
    qa, qb = a.get("retrieval_quality", {}), b.get("retrieval_quality", {})
    for mk in ["hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "recall@10", "precision@5", "mrr"]:
        print(f"    {mk:<12} {_fmt_delta(qa.get(mk,0), qb.get(mk,0), pct=True)}")

    # ── 延迟 & 3s 预算 ──
    print("\n  延迟 (ms, 越低越好) & 3s 达标率:")
    la, lb = a.get("latency", {}), b.get("latency", {})
    for mk in ["retrieval_avg", "llm_avg", "total_avg", "total_p50", "total_p95"]:
        print(f"    {mk:<14} {_fmt_delta(la.get(mk,0), lb.get(mk,0), lower_better=True)}")
    print(f"    under_3s_rate {_fmt_delta(la.get('under_3s_rate',0), lb.get('under_3s_rate',0), pct=True)}")

    # ── 逐题变化（通过状态翻转 / MRR 显著变化）──
    ra = {r["id"]: r for r in a.get("results", [])}
    rb = {r["id"]: r for r in b.get("results", [])}
    flips = []
    for qid in sorted(set(ra) & set(rb)):
        pa, pb = ra[qid].get("pass"), rb[qid].get("pass")
        mrr_a = ra[qid].get("retrieval", {}).get("mrr", 0)
        mrr_b = rb[qid].get("retrieval", {}).get("mrr", 0)
        if pa != pb or abs(mrr_a - mrr_b) >= 0.2:
            tag = "✅修复" if (not pa and pb) else ("❌回退" if (pa and not pb) else "〜检索变化")
            flips.append((qid, tag, pa, pb, mrr_a, mrr_b, ra[qid].get("focus", "")))
    if flips:
        print("\n  逐题变化 (通过翻转 或 MRR 变化≥0.2):")
        for qid, tag, pa, pb, ma, mb, focus in flips:
            print(f"    [{qid:>4}] {tag:<8} ({focus})  pass {pa}→{pb}  mrr {ma:.2f}→{mb:.2f}")
    else:
        print("\n  逐题：无通过翻转、无显著检索变化")
    print("=" * 78)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--list":
        runs = _list_runs()
        if not runs:
            print(f"无历史结果，目录: {_history_dir()}")
            return
        print(f"历史评估结果 ({len(runs)} 条) @ {_history_dir()}:")
        for r in runs:
            try:
                d = _load(r)
                print(f"  {os.path.basename(r)}  通过率={d.get('answer_pass_rate',0)*100:.0f}%  "
                      f"Hit@5={d.get('retrieval_quality',{}).get('hit@5',0):.2f}  "
                      f"<3s={d.get('latency',{}).get('under_3s_rate',0)*100:.0f}%")
            except Exception:
                print(f"  {os.path.basename(r)}  (读取失败)")
        return

    if len(args) >= 2:
        path_a, path_b = args[0], args[1]
    else:
        runs = _list_runs()
        if len(runs) < 2:
            print(f"历史结果不足 2 条（当前 {len(runs)} 条），无法对比。")
            print(f"目录: {_history_dir()}")
            print("先多跑几次 python eval_quick.py 生成历史。")
            return
        path_a, path_b = runs[-2], runs[-1]

    compare(path_a, path_b)


if __name__ == "__main__":
    main()
