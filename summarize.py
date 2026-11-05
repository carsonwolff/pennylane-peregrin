"""Summarize a results folder into markdown tables (and a plot if matplotlib is installed).

  python summarize.py results/20261104-210000
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(folder):
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]
    env = json.loads((folder / "env.json").read_text())
    return rows, env


def fmt_rate(r):
    if r is None:
        return "-"
    if r["status"] != "ok":
        return r["status"]
    s = f"{r['circuits_per_s']:,.0f}"
    return s if r.get("correct", True) else s + " (!)"


def tables(rows):
    by = {(r["backend"], r["n_qubits"], r["batch"]): r for r in rows}
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    qubits = sorted({r["n_qubits"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    out = []
    for n in qubits:
        out.append(f"\n### {n} qubits: throughput (circuits/s, higher is better)\n")
        out.append("| backend | " + " | ".join(f"B={b:,}" for b in batches) + " | compile/first call (largest B) |")
        out.append("|---|" + "---:|" * (len(batches) + 1))
        best = {b: max((by[k]["circuits_per_s"] for k in by if k[1] == n and k[2] == b
                        and by[k]["status"] == "ok"), default=None) for b in batches}
        for be in backends:
            cells = []
            for b in batches:
                r = by.get((be, n, b))
                c = fmt_rate(r)
                if r and r["status"] == "ok" and best[b] and r["circuits_per_s"] == best[b]:
                    c = f"**{c}**"
                cells.append(c)
            oks = [by[(be, n, b)] for b in batches if (be, n, b) in by and by[(be, n, b)]["status"] == "ok"]
            first = f"{oks[-1]['first_call_s']:.2f} s" if oks else "-"
            out.append(f"| {be} | " + " | ".join(cells) + f" | {first} |")
    bad = [r for r in rows if r["status"] == "ok" and not r.get("correct", True)]
    errs = [r for r in rows if r["status"] in ("error", "crash")]
    if bad:
        out.append("\n**Correctness mismatches (marked (!)):** " + ", ".join(
            f"{r['backend']} n={r['n_qubits']} (max|err| {r['max_abs_err']:.1e})" for r in bad))
    if errs:
        out.append("\n**Failures:**")
        for r in errs:
            out.append(f"- {r['backend']} n={r['n_qubits']} B={r['batch']}: {r.get('error', '')[:200]}")
    return "\n".join(out)


def plot(rows, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    qubits = sorted({r["n_qubits"] for r in rows})
    fig, axes = plt.subplots(1, len(qubits), figsize=(4.2 * len(qubits), 4), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], qubits):
        series = defaultdict(list)
        for r in rows:
            if r["n_qubits"] == n and r["status"] == "ok":
                series[r["backend"]].append((r["batch"], r["circuits_per_s"]))
        for be, pts in series.items():
            pts.sort()
            ax.plot(*zip(*pts), marker="o", label=be)
        ax.set(xscale="log", yscale="log", title=f"{n} qubits", xlabel="batch size")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("circuits / s")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = folder / "throughput.png"
    fig.savefig(path, dpi=150)
    return path


def main():
    folder = Path(sys.argv[1])
    rows, env = load(folder)
    hdr = (f"# Benchmark {env['timestamp']}\n\n"
           f"- CPU: {env['cpu']} ({env['cpu_count']} threads); GPU: {env.get('gpu') or 'none'}\n"
           f"- Precision: {env['args']['precision']}; layers: {env['args']['layers']}; "
           f"embedding: {env['args'].get('embed', 'amplitude')}; "
           f"OMP_NUM_THREADS: {env['env'].get('OMP_NUM_THREADS') or 'unset'}\n"
           f"- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()) + "\n"
           f"- Timing: median of steady-state reps, numpy in to numpy out; "
           f"first call (compile) reported separately.\n")
    md = hdr + tables(rows)
    (folder / "summary.md").write_text(md)
    print(md)
    p = plot(rows, folder)
    if p:
        print(f"\nplot: {p}")


if __name__ == "__main__":
    main()
