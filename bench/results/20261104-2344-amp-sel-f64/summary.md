# Benchmark 2026-11-05T01:38:01

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: amp-sel; precision: f64; layers: 3; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 5 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 11,658 | 36,275 | 34,476 | 31,479 | 3.09 s |
| dq-jax-cpu | 1,506 | 8,690 | 20,322 | 17,060 | 6.61 s |
| jax-hand-cpu | **186,658** | **463,168** | **301,104** | **346,605** | 1.08 s |
| lightning.qubit | 2,004 | 2,096 | 2,095 | 2,062 | 48.31 s |
| lightning.qubit+qjit | 10,362 | 11,222 | 10,943 | 10,809 | 9.82 s |
| qulacs | 485 | 481 | 483 | skipped | 20.85 s |