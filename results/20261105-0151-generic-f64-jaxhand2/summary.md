# Benchmark 2026-11-05T01:51:52

- CPU: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (4 threads); GPU: none
- Circuit: generic; precision: f64; layers: 1; embedding: stateprep; OMP_NUM_THREADS: unset
- Packages: numpy 2.5.4, scipy 1.18.1, pennylane 0.45.1, pennylane_lightning 0.45.0, pennylane_catalyst 0.15.0, jax 0.7.1, jaxlib 0.7.1, qulacs 0.6.14, pennylane_qulacs 0.44.0
- Timing: median of steady-state reps, numpy in to numpy out; first call (compile) reported separately.

### 2 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| jax-hand-cpu | **2,521,496** | **9,665,151** | **24,582,527** | **18,003,839** | 0.25 s |

### 4 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| jax-hand-cpu | **1,214,941** | **2,767,078** | **2,389,960** | **1,208,207** | 0.56 s |

### 6 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| jax-hand-cpu | **303,571** | **717,392** | **304,136** | **227,909** | 1.09 s |

### 8 qubits: throughput (circuits/s, higher is better)

| backend | B=100 | B=1,000 | B=10,000 | B=100,000 | compile/first call (largest B) |
|---|---:|---:|---:|---:|---:|
| jax-hand-cpu | **97,296** | **56,176** | **44,668** | **42,021** | 3.52 s |