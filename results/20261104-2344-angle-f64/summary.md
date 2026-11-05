# Benchmark 2026-11-05T01:33:41

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: angle-trainable; precision: f64; layers: 1; embedding: angle; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| default.qubit | 25,598 | 148,783 | 187,762 | 224,942 | 0.48 s |
| dq-jax-cpu | 6,441 | 47,952 | 174,559 | 202,353 | 1.78 s |
| jax-hand-cpu | **302,266** | **599,147** | **567,071** | **504,070** | 0.66 s |
| lightning.qubit | 5,132 | 5,549 | 5,055 | 5,150 | 19.49 s |
| lightning.qubit+qjit | 14,112 | 15,722 | 15,243 | 13,828 | 7.97 s |
| qulacs | 1,231 | 1,236 | 1,190 | skipped | 8.44 s |