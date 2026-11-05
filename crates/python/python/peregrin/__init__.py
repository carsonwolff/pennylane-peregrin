"""peregrin: fast batched simulation of many small circuits on the CPU.

    import peregrin
    c = peregrin.Circuit({
        "n_qubits": 2,
        "ops": [{"name": "RY", "wires": [0], "params": [["x", 0, 1.0]]},
                {"name": "CNOT", "wires": [0, 1]}],
        "measurements": [{"type": "expval", "word": [[1, "Z"]]}],
    })
    out = c.execute(X)          # X: (batch, d_x); out: (batch, n_outputs)

Params are ["x", col, scale] (column of the per-circuit inputs), ["w", idx, scale] (shared
weight) or ["c", value]. Wire 0 is the most significant bit, as in PennyLane.
"""
import json

import numpy as np

from ._core import __version__, get_num_threads, set_num_threads
from ._core import Circuit as _Circuit

__all__ = ["Circuit", "set_num_threads", "get_num_threads", "__version__"]


class Circuit:
    """A circuit template parsed once; execute() runs it for every row of a batch."""

    def __init__(self, spec):
        self.spec = spec if isinstance(spec, dict) else json.loads(spec)
        self._c = _Circuit(json.dumps(self.spec))

    @property
    def n_qubits(self):
        return self._c.n_qubits

    @property
    def n_outputs(self):
        return self._c.n_outputs

    def execute(self, inputs=None, weights=None, dtype=None):
        """inputs: (batch, d_x) array; weights: any shape, flattened. Returns (batch, n_outputs).

        dtype is float64 or float32; by default float32 when inputs are float32, else float64.
        """
        if inputs is None:
            raise ValueError("inputs is required; pass an array of shape (batch, 0) for no per-circuit data")
        if dtype is None:
            dtype = np.float32 if np.asarray(inputs).dtype == np.float32 else np.float64
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float64):
            raise TypeError("dtype must be float32 or float64")
        x = np.ascontiguousarray(inputs, dtype=dtype)
        if x.ndim != 2:
            raise ValueError(f"inputs must be 2-D (batch, d_x), got shape {x.shape}")
        w = np.ascontiguousarray(np.zeros(0) if weights is None else weights, dtype=dtype).reshape(-1)
        return self._c.execute(x, w)
