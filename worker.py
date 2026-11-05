"""Run one (backend, workload) case in a fresh process and print one JSON line.

Usage (normally invoked by run_bench.py):
  python worker.py '{"backend": "dq-jax-cpu", "n_qubits": 4, "batch": 1000, ...}'
"""
import json
import statistics
import sys
import time
import traceback

import numpy as np


def main(cfg):
    from backends import BACKENDS
    from workload import Workload, make_features, make_weights

    wl = Workload(cfg["n_qubits"], cfg["batch"], cfg.get("n_layers", 1), cfg.get("seed", 0),
                  cfg.get("embed", "stateprep"))
    precision = cfg.get("precision", "f64")
    reps, budget = cfg.get("reps", 10), cfg.get("budget_s", 60.0)
    check_rows = cfg.get("check_rows", 64)

    builder = BACKENDS[cfg["backend"]][0]
    in_dtype = np.float32 if precision == "f32" else np.float64
    F, W = make_features(wl, in_dtype), make_weights(wl, in_dtype)

    t = time.perf_counter()
    run, info = builder(wl, precision)
    t_build = time.perf_counter() - t

    # First call: includes tracing / JIT / qjit compilation.
    t = time.perf_counter()
    out = run(F, W)
    t_first = time.perf_counter() - t

    out = np.asarray(out)
    if out.shape != (wl.batch, wl.n_out):
        raise ValueError(f"bad output shape {out.shape}, expected {(wl.batch, wl.n_out)}")
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite outputs")

    # Steady state: at least 3 reps (or 1 if a single call blows the budget).
    times, spent = [], 0.0
    while len(times) < reps:
        t = time.perf_counter()
        run(F, W)
        dt = time.perf_counter() - t
        times.append(dt)
        spent += dt
        if spent > budget and len(times) >= min(3, reps):
            break

    med = statistics.median(times)
    return {
        "status": "ok",
        **info,
        "build_s": t_build,
        "first_call_s": t_first,
        "median_s": med,
        "min_s": min(times),
        "mean_s": statistics.fmean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "n_reps": len(times),
        "circuits_per_s": wl.batch / med,
        "us_per_circuit": 1e6 * med / wl.batch,
        "check": out[:check_rows].astype(np.float64).tolist(),
    }


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    try:
        res = main(cfg)
    except Exception as e:  # report, don't crash the sweep
        res = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    print("@@RESULT@@" + json.dumps(res))
