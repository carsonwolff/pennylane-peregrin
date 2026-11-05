# pennylane-peregrin

Fast CPU simulation of large batches of small quantum circuits (1 to 10 qubits), with a
PennyLane device. It is built for quanvolutional preprocessing: one circuit per image patch,
hundreds of thousands of patches per call. The simulator core is written in Rust.

```bash
pip install pennylane-peregrin
```

```python
import numpy as np
import pennylane as qml

dev = qml.device("peregrin.cpu", wires=4)    # precision="f32" and threads=N are optional

@qml.qnode(dev)
def quanv(patches, weights):                 # patches: (batch, 4), one row per image patch
    for i in range(4):
        qml.RY(np.pi * patches[:, i], wires=i)
    qml.RandomLayers(weights, wires=range(4))
    return [qml.expval(qml.Z(i)) for i in range(4)]

out = quanv(np.random.rand(100_000, 4), np.random.uniform(0, 2 * np.pi, (1, 4)))
```

Write the circuit for the whole batch with parameter broadcasting, so that the whole batch
runs as one call. Outputs match `default.qubit` to 1e-12 in f64. On a 2-core laptop it was
faster than every other PennyLane CPU device we measured, at every qubit count and batch size
we tested: 5.3x to 73x at 100k circuits. Benchmarks, methodology, and the direct
`peregrin.Circuit` API (no PennyLane overhead) are in the
[repository](https://github.com/carsonwolff/pennylane-peregrin).

Supported:
- **Gates:** any gate PennyLane decomposes to peregrin's gate set.
- **State preparation:** StatePrep, AmplitudeEmbedding or BasisState as the first operation.
- **Measurements:** expval and var of Pauli words and Hamiltonians, and of matrix observables
  on up to 4 wires; probs; state.
- **Gradients:** parameter-shift.

Not yet supported: shots, mid-circuit measurements, and device gradients.

License: MIT OR Apache-2.0.
