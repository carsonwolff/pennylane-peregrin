"""Compare one backend against the fastest other backend in every cell.

    python compare.py results/<run-a> results/<run-b> ... [--ours peregrin-cpu] [--exclude b ...] [--md]

A cell is (circuit, precision, embed, n_qubits, batch). Rows from all given folders are
pooled; when one backend has several rows for a cell (a re-run), the newest folder wins,
which is the latest timestamp in the folder name. Only rows that passed the correctness
check count. Prints, per cell: our circuits/s, the best other backend and its circuits/s,
and the ratio.
"""
import argparse
import json
from pathlib import Path


def load(dirs):
    rows = {}
    for d in sorted(dirs, key=lambda p: Path(p).name):  # older first, newer overwrite
        for line in open(Path(d) / "results.jsonl"):
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("correct"):
                continue
            key = (r["circuit"], r["precision"], r.get("embed", "stateprep"), r["n_qubits"], r["batch"])
            rows[key + (r["backend"],)] = r["circuits_per_s"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ours", default="peregrin-cpu")
    ap.add_argument("--exclude", nargs="*", default=[], help="backends to leave out of 'best other'")
    ap.add_argument("--md", action="store_true", help="markdown table")
    args = ap.parse_args()
    rows = load(args.dirs)
    cells = sorted({k[:5] for k in rows})
    out = []
    for cell in cells:
        ours = rows.get(cell + (args.ours,))
        others = {k[5]: v for k, v in rows.items()
                  if k[:5] == cell and k[5] != args.ours and k[5] not in args.exclude}
        if ours is None or not others:
            continue
        best = max(others, key=others.get)
        out.append((cell, ours, best, others[best], ours / others[best]))
    if args.md:
        print(f"| circuit | prec | embed | n | B | {args.ours} | best other | its circuits/s | ratio |")
        print("|---|---|---|---:|---:|---:|---|---:|---:|")
        for (c, p, e, n, b), o, best, bv, ratio in out:
            print(f"| {c} | {p} | {e} | {n} | {b:,} | {o:,.0f} | {best} | {bv:,.0f} | {ratio:.2f}x |")
    else:
        for (c, p, e, n, b), o, best, bv, ratio in out:
            print(f"{c:16} {p} {e:9} n={n} B={b:<7} {o:>13,.0f}  vs {best:22} {bv:>13,.0f}  {ratio:6.2f}x")
    if out:
        worst = min(out, key=lambda t: t[4])
        print(f"\nslowest ratio: {worst[4]:.2f}x at {worst[0]} against {worst[2]}; "
              f"{sum(t[4] > 1 for t in out)} of {len(out)} cells faster")


if __name__ == "__main__":
    main()
