//! Batch entry points. The batch is split into chunks across the current rayon pool; each
//! chunk reuses one pair of state buffers for all its circuits, so nothing is allocated per gate
//! or per circuit.

use num_complex::Complex;
use rayon::prelude::*;

use crate::circuit::Circuit;
use crate::program::{probs_map, Program, Scratch};
use crate::rng::SplitMix64;
use crate::tile::{TileProgram, MAX_TILE_QUBITS};
use crate::{err, Real, Result};

/// Below this many circuits the batch runs on the calling thread.
const SERIAL_BELOW: usize = 256;

fn chunk_rows(batch: usize) -> usize {
    let tasks = rayon::current_num_threads().max(1) * 8;
    batch.div_ceil(tasks).clamp(64, 8192)
}

/// Per-circuit input width implied by the buffer length.
fn d_x_of<T>(batch: usize, inputs: &[T]) -> Result<usize> {
    if batch == 0 {
        Ok(0)
    } else if inputs.len() % batch != 0 {
        err(format!("inputs length {} is not a multiple of batch {batch}", inputs.len()))
    } else {
        Ok(inputs.len() / batch)
    }
}

/// Run `batch` copies of `circuit`. `inputs` is row-major `(batch, d_x)` per-circuit data
/// (d_x may be 0); `weights` is shared by every circuit. Returns row-major
/// `(batch, circuit.n_outputs())`.
pub fn execute<T: Real>(circuit: &Circuit, batch: usize, inputs: &[T], weights: &[T]) -> Result<Vec<T>> {
    let mut out = vec![T::zero(); batch * circuit.n_outputs()];
    execute_into(circuit, batch, inputs, weights, &mut out)?;
    Ok(out)
}

/// [`execute`] into a caller-provided `(batch, n_outputs)` buffer.
pub fn execute_into<T: Real>(circuit: &Circuit, batch: usize, inputs: &[T], weights: &[T], out: &mut [T]) -> Result<()> {
    let d_x = d_x_of(batch, inputs)?;
    let prog = Program::new(circuit, d_x, weights)?;
    let w = prog.n_out;
    if out.len() != batch * w {
        return err(format!("output buffer has {} values, need {}", out.len(), batch * w));
    }
    if batch == 0 || w == 0 {
        return Ok(());
    }
    let tiled = (circuit.n_qubits <= MAX_TILE_QUBITS).then(|| TileProgram::new(&prog));
    let run = |rows: &[T], dst: &mut [T]| {
        if let Some(tp) = &tiled {
            return tp.run(rows, d_x.max(1), dst);
        }
        let mut sc = Scratch::new(prog.dim);
        for (x, o) in rows.chunks_exact(d_x.max(1)).zip(dst.chunks_exact_mut(w)) {
            prog.evolve(x, &mut sc);
            prog.measure(&sc.psi, o);
        }
    };
    if d_x == 0 {
        // No per-circuit data: every row is the same circuit. Still compute each row, because
        // that is what was asked for, but feed an empty input slice.
        let dummy = vec![T::zero(); batch];
        return execute_rows(batch, w, &dummy, 1, out, &run);
    }
    execute_rows(batch, w, inputs, d_x, out, &run)
}

fn execute_rows<T: Real, F>(batch: usize, w: usize, inputs: &[T], d_x: usize, out: &mut [T], run: &F) -> Result<()>
where
    F: Fn(&[T], &mut [T]) + Sync,
{
    if batch < SERIAL_BELOW {
        run(inputs, out);
        return Ok(());
    }
    let rows = chunk_rows(batch);
    out.par_chunks_mut(rows * w)
        .zip(inputs.par_chunks(rows * d_x))
        .for_each(|(dst, src)| run(src, dst));
    Ok(())
}

/// Final state vectors, row-major `(batch, 2^n)`.
pub fn final_states<T: Real>(circuit: &Circuit, batch: usize, inputs: &[T], weights: &[T]) -> Result<Vec<Complex<T>>> {
    let d_x = d_x_of(batch, inputs)?;
    let prog = Program::new(circuit, d_x, weights)?;
    let dim = prog.dim;
    let mut out = vec![Complex::new(T::zero(), T::zero()); batch * dim];
    let dummy;
    let (src, dxe) = if d_x == 0 {
        dummy = vec![T::zero(); batch];
        (&dummy[..], 1)
    } else {
        (inputs, d_x)
    };
    let rows = chunk_rows(batch.max(1));
    out.par_chunks_mut(rows * dim).zip(src.par_chunks(rows * dxe)).for_each(|(dst, xs)| {
        let mut sc = Scratch::new(dim);
        for (x, o) in xs.chunks_exact(dxe).zip(dst.chunks_exact_mut(dim)) {
            prog.evolve(x, &mut sc);
            o.copy_from_slice(&sc.psi);
        }
    });
    Ok(out)
}

/// `shots` computational-basis samples per circuit on `wires`, as outcome indices with the
/// first wire most significant. Row-major `(batch, shots)`. Circuit `b` draws from its own
/// stream seeded by `(seed, b)`, so results do not depend on the thread count.
pub fn sample<T: Real>(
    circuit: &Circuit,
    batch: usize,
    inputs: &[T],
    weights: &[T],
    wires: &[usize],
    shots: usize,
    seed: u64,
) -> Result<Vec<u32>> {
    let d_x = d_x_of(batch, inputs)?;
    let prog = Program::new(circuit, d_x, weights)?;
    let n = circuit.n_qubits;
    if wires.is_empty() || wires.iter().any(|&w| w >= n) || wires.len() > 31 {
        return err("sample: wires must be non-empty and in range");
    }
    for (i, w) in wires.iter().enumerate() {
        if wires[..i].contains(w) {
            return err(format!("sample: repeated wire {w}"));
        }
    }
    let map = probs_map(n, wires);
    let k = 1usize << wires.len();
    let mut out = vec![0u32; batch * shots];
    if batch == 0 || shots == 0 {
        return Ok(out);
    }
    let dummy;
    let (src, dxe) = if d_x == 0 {
        dummy = vec![T::zero(); batch];
        (&dummy[..], 1)
    } else {
        (inputs, d_x)
    };
    let rows = chunk_rows(batch);
    out.par_chunks_mut(rows * shots)
        .zip(src.par_chunks(rows * dxe))
        .enumerate()
        .for_each(|(ci, (dst, xs))| {
            let mut sc = Scratch::new(prog.dim);
            let mut cum = vec![0.0f64; k];
            for (r, (x, o)) in xs.chunks_exact(dxe).zip(dst.chunks_exact_mut(shots)).enumerate() {
                prog.evolve(x, &mut sc);
                Program::marginal(&sc.psi, &map, &mut cum);
                let mut acc = 0.0;
                for c in cum.iter_mut() {
                    acc += *c;
                    *c = acc;
                }
                let mut rng = SplitMix64::for_circuit(seed, (ci * rows + r) as u64);
                for s in o.iter_mut() {
                    let u = rng.next_f64() * acc;
                    *s = cum.partition_point(|&c| c <= u).min(k - 1) as u32;
                }
            }
        });
    Ok(out)
}
