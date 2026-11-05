"""Benchmark sweep: every backend x qubit count x batch size, each case in its own process.

Examples
  python run_bench.py --list
  python run_bench.py                                  # all available backends, default sweep
  python run_bench.py --backends dq-jax-gpu jax-hand-gpu lightning.gpu+qjit --qubits 4 8
  python run_bench.py --batches 100 1000 10000 100000 --precision f32 --threads 8
  python run_bench.py --embed amplitude --qubits 4 8     # the paper's AmplitudeEmbedding path

Output: results/<timestamp>/{env.json, results.jsonl}; then run summarize.py on that folder.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backends import BACKENDS, REFERENCE  # noqa: E402

PKGS = [
    "numpy", "scipy", "pennylane", "pennylane_lightning", "pennylane_lightning_gpu",
    "pennylane_catalyst", "jax", "jaxlib", "jax_cuda12_plugin", "jax_cuda13_plugin",
    "qulacs", "qulacs_gpu", "pennylane_qulacs", "custatevec_cu12", "custatevec_cu13",
]

REQUIRES = {  # import names or "device:<pennylane device>" needed per backend
    "default.qubit": ["pennylane"],
    "dq-jax-cpu": ["pennylane", "jax"], "dq-jax-gpu": ["pennylane", "jax"],
    "jax-hand-cpu": ["jax"], "jax-hand-gpu": ["jax"],
    "lightning.qubit": ["device:lightning.qubit"],
    "lightning.qubit+qjit": ["device:lightning.qubit", "catalyst"],
    "lightning.gpu": ["device:lightning.gpu"],
    "lightning.gpu+qjit": ["device:lightning.gpu", "catalyst"],
    "qulacs": ["device:qulacs.simulator", "qulacs"],
    "qulacs-gpu": ["device:qulacs.simulator", "qulacs"],
}


def has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _pl_devices():
    from importlib.metadata import entry_points

    return {ep.name for ep in entry_points(group="pennylane.plugins")} | {"default.qubit"}


def available(name, gpu):
    if BACKENDS[name][2] and not gpu:
        return False
    devs = _pl_devices()
    for req in REQUIRES[name]:
        if req.startswith("device:"):
            if req[7:] not in devs:
                return False
        elif importlib.util.find_spec(req) is None:
            return False
    return True


def collect_env(args):
    from importlib import metadata

    versions = {}
    for p in PKGS:
        try:
            versions[p] = metadata.version(p.replace("_", "-"))
        except metadata.PackageNotFoundError:
            pass
    env = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES"]},
        "args": vars(args),
    }
    if shutil.which("nvidia-smi"):
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True, text=True)
        env["gpu"] = q.stdout.strip()
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                                           text=True).stdout.strip() or None
    except FileNotFoundError:
        env["git_commit"] = None
    return env


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_case(backend, cfg, timeout, threads):
    env = dict(os.environ)
    env.update(BACKENDS[backend][1])
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't grab all GPU memory
    try:
        p = subprocess.run([sys.executable, str(HERE / "worker.py"), json.dumps({**cfg, "backend": backend})],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout}s"}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"status": "crash", "error": f"exit code {p.returncode}", "stderr": p.stderr[-3000:]}


def max_abs_err(a, b):
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=["all"])
    ap.add_argument("--qubits", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--batches", nargs="+", type=int, default=[100, 1000, 10000])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--embed", choices=["stateprep", "amplitude"], default="stateprep",
                    help="state loading; 'amplitude' reproduces AmplitudeEmbedding's decomposition cost under qjit/qulacs")
    ap.add_argument("--precision", choices=["f64", "f32"], default="f64")
    ap.add_argument("--reps", type=int, default=10, help="timed repetitions after the warm-up call")
    ap.add_argument("--budget", type=float, default=60.0, help="soft time budget (s) for the timed reps of one case")
    ap.add_argument("--timeout", type=float, default=900.0, help="hard kill (s) per case, compile included")
    ap.add_argument("--threads", type=int, default=0, help="set OMP_NUM_THREADS for workers (0 = leave as is)")
    ap.add_argument("--tol", type=float, default=None, help="max |error| vs reference (default 1e-9 f64, 1e-4 f32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="show backends and whether they are available")
    args = ap.parse_args()

    gpu = has_gpu()
    if args.list:
        for name in BACKENDS:
            print(f"{name:22s} {'available' if available(name, gpu) else 'missing'}")
        return

    names = list(BACKENDS) if args.backends == ["all"] else args.backends
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        sys.exit(f"unknown backends: {unknown}. Use --list.")
    skipped = [n for n in names if not available(n, gpu)]
    names = [n for n in names if n not in skipped]
    if skipped:
        print(f"skipping unavailable: {', '.join(skipped)}")
    tol = args.tol if args.tol is not None else (1e-9 if args.precision == "f64" else 1e-4)

    out_dir = args.out or HERE / "results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(collect_env(args), indent=2, default=str))
    results_path = out_dir / "results.jsonl"
    print(f"writing to {out_dir}")

    base = {"n_layers": args.layers, "seed": args.seed, "embed": args.embed, "reps": args.reps, "budget_s": args.budget}

    # Reference outputs (default.qubit, f64) on the first 64 circuits, per qubit count.
    refs = {}
    for n in args.qubits:
        r = run_case(REFERENCE, {**base, "n_qubits": n, "batch": 64, "precision": "f64", "reps": 1},
                     args.timeout, args.threads)
        if r["status"] != "ok":
            sys.exit(f"reference failed for n={n}: {r.get('error')}\n{r.get('traceback', r.get('stderr', ''))}")
        refs[n] = r["check"]

    with results_path.open("a") as fh:
        for backend in names:
            for n in args.qubits:
                prev = None  # (batch, median_s) of last successful case, for projection
                for B in sorted(args.batches):
                    cfg = {**base, "n_qubits": n, "batch": B, "precision": args.precision}
                    if prev and prev[1] * B / prev[0] > args.timeout:
                        res = {"status": "skipped", "error": "projected to exceed timeout"}
                    else:
                        res = run_case(backend, cfg, args.timeout, args.threads)
                    if res["status"] == "ok":
                        k = min(len(res["check"]), len(refs[n]))
                        err = max_abs_err(res["check"][:k], refs[n][:k])
                        res["max_abs_err"] = err
                        res["correct"] = err <= tol
                        prev = (B, res["median_s"])
                    res.pop("check", None)
                    row = {"backend": backend, **cfg, **res}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    _print_row(row)
    print(f"\ndone. summarize with:  python summarize.py {out_dir}")


def _print_row(r):
    head = f"{r['backend']:22s} n={r['n_qubits']} B={r['batch']:<7d}"
    if r["status"] != "ok":
        print(f"{head} {r['status'].upper()}: {r.get('error', '')[:120]}")
        return
    flag = "" if r["correct"] else f"  MISMATCH max|err|={r['max_abs_err']:.2e}"
    print(f"{head} {r['circuits_per_s']:>12,.0f} circ/s  median {r['median_s'] * 1e3:9.2f} ms  "
          f"first {r['first_call_s']:7.2f} s{flag}")


if __name__ == "__main__":
    main()
