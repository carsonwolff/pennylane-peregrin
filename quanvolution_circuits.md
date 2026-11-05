# Quanvolutional circuits in recent papers

Reference for the PennyLane device project. Collected 2026-11-04. It covers what circuits recent quanvolutional and closely related papers actually run, and what that means for the qbench workloads and the Rust device's gate set.

Details marked "not stated" were missing from the paper or the page read. Do not fill them in by guessing. Read the PDF first.

## At a glance

| Paper | Year | Qubits | Patch / window | Encoding | Processing circuit | Measurement | Sim |
|---|---|---|---|---|---|---|---|
| PennyLane quanvolution demo (Mari) | 2020, updated 2026 | 4 | 2x2, stride 2 | RY(pi x) | RandomLayers, 1 layer, fixed | <Z> x4 | default.qubit |
| Pneumonia QNN (Tanbhir et al.), arXiv 2510.23660 | 2025 | 4 | 2x2 (stride 2 implied) | RY(pi x) | RandomLayers, PennyLane defaults | <Z> x4 | PennyLane, backend not stated |
| Spectrum peak-finding (Bischof et al.), arXiv 2512.13125 | 2025 | 5 | 1D window of 32, stride 1 | Amplitude (32 values) | 3 ansatzes, 3 layers each, trainable | <Z> x5, then ReLU | lightning.qubit |
| Integrated encoding + quantization (Lizzio Bosco et al.), arXiv 2410.05777 | 2024 | 4 (Int) or k^2 (Rot) | 2x2 to 5x5 | RX(pi x), or integrated two-qubit Pauli rotations | Henderson random circuit, or none (integrated) | Z^n, 1000 shots, mean count of 1s | Qiskit 1.1 |
| ResQuNNs (Kashif, Shafique), arXiv 2402.09146 | 2024 | 4 | 2x2, stride 2 | RY(x) | RX(theta) per qubit + CNOT chain, trainable, stacked with residuals | <Z> x4 | PennyLane |
| Noisy HQNNs (Ahmed et al.), arXiv 2505.03378 | 2025 | 4 | 2x2 implied | RY(x) | 3 BasicEntanglerLayers, trainable | not stated | lightning.qubit, default.mixed |
| Brain tumor MRI QCNN (Nugraha et al.), arXiv 2509.02582 | 2025 | 4 | filter size 2 | not stated | depth 4, gates not stated | not stated | not stated |

The original design is Henderson et al., "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits", arXiv 1904.04767 (2019). It uses angle encoding with a fixed random circuit.

## Paper notes

### 1. PennyLane quanvolution demo (Andrea Mari)

This is the de facto reference implementation that most applied papers copy.

- 4 qubits, one per pixel of a 2x2 patch, stride 2, 28x28 MNIST in, 14x14x4 out.
- Encoding: `RY(pi * x_j)` on wire j, pixels in [0, 1].
- Processing: `qml.RandomLayers(rand_params, wires=range(4))` with `n_layers = 1` and `rand_params ~ U[0, 2pi)` of shape (1, 4). Weights are fixed (not trained). Seed 0.
- Measurement: `expval(PauliZ(j))` for j in 0..3, giving 4 channels.
- Device: `default.qubit`. n_train 50, n_test 30.

```python
dev = qml.device("default.qubit", wires=4)
rand_params = np.random.uniform(high=2 * np.pi, size=(1, 4))

@qml.qnode(dev)
def circuit(phi):
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)
    qml.RandomLayers(rand_params, wires=list(range(4)))
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]
```

Note: `RandomLayers` draws its gate layout from a seed. Fix it explicitly (`seed=` argument) in benchmarks so every backend runs the same gates.

### 2. Quanvolutional Neural Networks for Pneumonia Detection (Tanbhir, Shahriyar, Chy), arXiv 2510.23660, Oct 2025, NCIM 2025

- Essentially the PennyLane demo circuit applied to medical images: 4 qubits, 2x2 patches, `RY(pi x)` encoding, `RandomLayers` with PennyLane's default settings, `<Z>` on each qubit.
- 28x28 PneumoniaMNIST in, 14x14x4 out (so stride 2, though the paper does not say so).
- Number of random layers and whether they are trained: not stated (the text and Figure 2 disagree).
- Classical side: TensorFlow/Keras. Trained on only 50 samples.
- Closest published analogue to an MRI-style patch preprocessing pipeline.

### 3. Quanvolutional Neural Networks for Spectrum Peak-Finding (Bischof et al.), arXiv 2512.13125, Dec 2025

The only recent paper here that uses amplitude embedding, as the owner's MRI paper did.

- 5 qubits. 1D spectrum of 200 points, sliding window of 32, stride 1, giving 169 windows per spectrum.
- Encoding: amplitude embedding of the L2-normalized 32 values. The paper notes this is Mottonen state prep on hardware but can be set directly in simulation. This is the same `AmplitudeEmbedding` vs `StatePrep` issue qbench found.
- Three trainable ansatzes, all 3 layers:
  1. StronglyEntanglingLayers style: RX, RY, RZ on every qubit, then a CNOT chain.
  2. Simplified two-design: initial RY layer, then 3 x (CZ entangler, RY layer).
  3. Random: rotations and CNOTs placed at random, all rotations trainable.
- Measurement: `<Z>` on all 5 qubits, ReLU to [0, 1]. 5 channels x 169 positions, max-pool 5, then 165 features.
- PennyLane 0.32 + `lightning.qubit`, PyTorch 2.0. Gradients by parameter shift or adjoint. Noisy runs use SPSA.
- 919 to 1,150 spectra per dataset, so about 155k to 195k circuit evaluations per epoch.

### 4. Integrated Encoding and Quantization to Enhance Quanvolutional Neural Networks (Lizzio Bosco, Portelli, Serra), arXiv 2410.05777, Oct 2024

Two ideas directly relevant to throughput.

- **QNN-Rot (baseline):** one qubit per pixel, so 4, 9, or 16 qubits for 2x2, 3x3, 4x4 filters. Encoding `RX(pi x)`. Processing is the Henderson random circuit: up to 2k^2 single-qubit gates from {RX, RY, RZ, S, T, H} on random qubits, and each qubit pair gets a two-qubit gate with probability 0.15 from {CNOT, SWAP, SqrtSWAP}. 5x5 (25 qubits) was skipped as too expensive to simulate.
- **QNN-Int (integrated encoding):** always 4 qubits, any filter size. The circuit is L = 2k^2 gates of the form `exp(-i alpha(x) sigma_1 (x) sigma_2)` on two distinct qubits, sigma in {I, X, Y, Z}, each taking one randomly chosen pixel. alpha is one of: `x pi`, `2 beta x pi`, or `(beta x + s) pi`, with beta, s ~ U[0, 1] per gate. In PennyLane terms these are `qml.PauliRot(2 * alpha, "XZ", wires=[a, b])` style gates.
- **Quantization + memoization:** pixels quantized as `floor(x N) / (N - 1)` with N = 50. Outputs cached in a lookup table per unique quantized patch. For 3x3 patches this cut circuit executions by 96.6% (MiraBest) and 97.1% (LArTPC).
- Measurement: `Z^n` with 1000 shots per patch, output = mean number of qubits measured in |1>. 8 filters per layer.
- Qiskit 1.1, noiseless. Datasets: MiraBest and LArTPC (both 30x30, about 700 images), plus MNIST and CIFAR10 for the quantization analysis. Up to 603,680 3x3 patches before quantization.

### 5. ResQuNNs: Towards Enabling Deep Learning in Quantum Convolution Neural Networks (Kashif, Shafique), arXiv 2402.09146

Journal version titled "Deep quanvolutional neural networks with enhanced trainability and gradient propagation".

- 4 qubits per 2x2 patch, stride 2.
- Encoding `RY(x_i)`. Trainable `RX(theta_i)` on each qubit, then CNOT(j, j+1) for j = 0..2 (open chain, 3 CNOTs). The paper's motivating example instead uses 4 RY + 3 CNOT, so the gate choice is inconsistent within the paper.
- Stacked quanvolutional layers (2 and 3 deep) with residual skips. Only O1+O2 and (X+O1)+O2 style skips let gradients reach every layer.
- `<Z>` per qubit as output channels, re-encoded for the next layer. PennyLane, Adam lr 0.01, 30 epochs. MNIST 28x28, 1,600 train / 400 test.
- Relevant because trained, stacked layers need gradients through every circuit, which multiplies circuit count by the parameter-shift factor.

### 6. Noisy HQNNs: A Comprehensive Analysis of Noise Robustness in Hybrid Quantum Neural Networks (Ahmed, Marchisio, Kashif, Shafique), arXiv 2505.03378, 2025

- QuanNN: 4 qubits (2x2 filter), `RY` angle encoding, 3 trainable BasicEntanglerLayers (RX rotations + CNOT ring by default; connectivity not stated in the paper).
- Noise channels inserted after every rotation block and entangler. Noise-free on `lightning.qubit`, noisy on `default.mixed`.
- MNIST and Fashion-MNIST, first 4 classes, 500 train / 150 validation. Hardware: one RTX 6000 Ada.
- Relevant if the device ever supports density-matrix or noisy simulation.

### 7. Application of Quantum Convolutional Neural Networks for MRI-Based Brain Tumor Detection and Classification (Nugraha et al.), arXiv 2509.02582, Aug 2025

- Closest dataset to the owner's paper: 3,264 brain MRI images (glioma, meningioma, pituitary, none).
- The page read gives only: 4 qubits, depth 4, filter size 2, 10 epochs. Encoding, gates, and measurement not stated. Read the PDF before using it as a workload.

### Not quanvolutional, but often confused with it

"Leveraging Quantum-Based Architectures for Robust Diagnostics" (Sodagari, Long), arXiv 2511.12386, Nov 2025 is a QCNN (convolution and pooling layers acting across one register of 8 data + 4 ancilla qubits fed by ResNet50 features), not a patch filter. One circuit per image, not per patch, so it is not the target workload.

## Implications for the project

### Workloads worth adding to qbench

| Name | Qubits | Circuit | Why |
|---|---|---|---|
| `owner-mri` (current default) | 2, 4 | amplitude/StatePrep, RY+RZ, CNOT ring, <Z>+<X> | The owner's paper. Confirm exact gates. |
| `pennylane-demo` | 4 | RY(pi x) x4, RandomLayers (1 layer, fixed seed), <Z> x4 | What most applied papers run. |
| `angle-trainable` | 4 | RY(x) x4, RX(theta) x4, CNOT chain, <Z> x4 | ResQuNNs; needs gradients. |
| `amp-5q-sel` | 5 | amplitude (32), 3 x (RX RY RZ + CNOT chain), <Z> x5 | Spectrum paper; tests the 5-qubit and amplitude path. |
| `henderson-9q` | 9 | RX(pi x) x9, Henderson random circuit, Z^9 sampled | 3x3 filters; pushes toward the top of the 2-8 qubit target. |
| `integrated-4q` | 4 | 2k^2 two-qubit Pauli rotations, data in the angles | Different gate mix; one parameter per gate. |

### Gate set the device should run natively

- Single qubit: RX, RY, RZ, Rot, S, T, H, Paulis (RandomLayers and Henderson circuits draw from these).
- Two qubit: CNOT, CZ, SWAP, SqrtSWAP, and PauliRot / IsingXX/YY/ZZ style `exp(-i a P(x)Q)` gates.
- State prep: native StatePrep for amplitude embedding (never decompose).
- Templates to expand on the Python side before reaching Rust: RandomLayers, BasicEntanglerLayers, StronglyEntanglingLayers, SimplifiedTwoDesign.

### Measurements

- `<Z>` per qubit is universal. `<X>` also appears in the owner's paper.
- Shot-based sampling (1000 shots in the integrated-encoding paper) should be supported with a seed.
- Gradients matter for ResQuNNs, the spectrum paper, and Noisy HQNNs, which all train the filter. Fixed random filters (demo, pneumonia, Henderson) need forward passes only.

### Throughput notes

- Patch counts per run in these papers range from about 10k to 600k circuits, which matches the qbench batch sweep. 100k should be in the standard sweep.
- Quantization plus memoization (paper 4) is an algorithmic speedup that applies to any fixed filter. It is worth offering as an option in the plugin, but benchmark it separately from raw simulation speed.

## Sources

- [Quanvolutional Neural Networks for Pneumonia Detection, arXiv 2510.23660](https://arxiv.org/pdf/2510.23660v1)
- [Quanvolutional Neural Networks for Spectrum Peak-Finding, arXiv 2512.13125](https://arxiv.org/html/2512.13125v1)
- [Integrated Encoding and Quantization to Enhance Quanvolutional Neural Networks, arXiv 2410.05777](https://arxiv.org/html/2410.05777v1)
- [ResQuNNs, arXiv 2402.09146](https://arxiv.org/pdf/2402.09146v6)
- [Noisy HQNNs, arXiv 2505.03378](https://arxiv.org/html/2505.03378v1)
- [Application of QCNNs for MRI-Based Brain Tumor Detection, arXiv 2509.02582](https://arxiv.org/abs/2509.02582v1)
- [Leveraging Quantum-Based Architectures for Robust Diagnostics, arXiv 2511.12386](https://arxiv.org/pdf/2511.12386v1)
- [PennyLane quanvolution demo](https://www.pennylane.ai/qml/demos/tutorial_quanvolution)
- [Emergent Mind topic overview: quanvolutional neural networks](https://www.emergentmind.com/topics/quanvolutional-neural-networks-quanvnns)
