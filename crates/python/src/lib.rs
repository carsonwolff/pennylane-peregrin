//! Python bindings: `peregrin._core`. The public Python API lives in python/peregrin.

use std::sync::{Arc, Mutex};

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use peregrin_core::{execute_into, Circuit as CoreCircuit, Real, SERIAL_BELOW};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

static POOL: Mutex<Option<Arc<rayon::ThreadPool>>> = Mutex::new(None);

fn pool() -> Option<Arc<rayon::ThreadPool>> {
    POOL.lock().unwrap().clone()
}

/// Use `n` worker threads for every later call (0 restores the default: one per logical CPU,
/// or RAYON_NUM_THREADS).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    let p = if n == 0 {
        None
    } else {
        Some(Arc::new(
            rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| PyValueError::new_err(e.to_string()))?,
        ))
    };
    *POOL.lock().unwrap() = p;
    Ok(())
}

#[pyfunction]
fn get_num_threads() -> usize {
    match pool() {
        Some(p) => p.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

/// A circuit template, parsed once from JSON (see peregrin_core's json module).
#[pyclass(frozen, module = "peregrin._core")]
struct Circuit {
    c: CoreCircuit,
}

fn run<'py, T: Real + numpy::Element>(
    py: Python<'py>,
    c: &CoreCircuit,
    x: PyReadonlyArray2<'py, T>,
    w: PyReadonlyArray1<'py, T>,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let batch = x.shape()[0];
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("inputs must be C-contiguous"))?;
    let ws = w.as_slice().map_err(|_| PyValueError::new_err("weights must be contiguous"))?;
    let out = PyArray2::<T>::zeros(py, [batch, c.n_outputs()], false);
    {
        let mut guard = out.readwrite();
        let dst = guard.as_slice_mut().expect("fresh array is contiguous");
        // Small batches run serially on this thread; handing them to the pool costs more than
        // the work.
        let p = if batch < SERIAL_BELOW { None } else { pool() };
        py.allow_threads(|| match p {
            Some(p) => p.install(|| execute_into(c, batch, xs, ws, dst)),
            None => execute_into(c, batch, xs, ws, dst),
        })
        .map_err(|e| PyValueError::new_err(e.0))?;
    }
    Ok(out)
}

#[pymethods]
impl Circuit {
    #[new]
    fn new(spec_json: &str) -> PyResult<Self> {
        CoreCircuit::from_json(spec_json).map(|c| Circuit { c }).map_err(|e| PyValueError::new_err(e.0))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.c.n_qubits
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.c.n_outputs()
    }

    /// Run the circuit once per row of `inputs` (batch, d_x) with shared `weights` (k,).
    /// Both must have the same dtype, float64 or float32; the result has that dtype too.
    fn execute<'py>(&self, py: Python<'py>, inputs: &Bound<'py, PyAny>, weights: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let (Ok(x), Ok(w)) = (inputs.extract::<PyReadonlyArray2<f64>>(), weights.extract::<PyReadonlyArray1<f64>>()) {
            return Ok(run(py, &self.c, x, w)?.into_any().unbind());
        }
        if let (Ok(x), Ok(w)) = (inputs.extract::<PyReadonlyArray2<f32>>(), weights.extract::<PyReadonlyArray1<f32>>()) {
            return Ok(run(py, &self.c, x, w)?.into_any().unbind());
        }
        Err(PyTypeError::new_err("inputs (2-D) and weights (1-D) must be numpy arrays of the same dtype, float64 or float32"))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
