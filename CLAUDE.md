# kantorovich

High-performance optimal transport library in Rust with Python bindings (PyO3).

## Build

```bash
# Development (debug, fast compile)
maturin develop

# Release (optimized, benchmarks)
RUSTFLAGS="-C target-cpu=native" maturin develop --release

# Run Rust tests
cargo test

# Run benchmarks
python benchmarks/bench_vs_pot.py
```

## Architecture

- `src/sinkhorn.rs` — Sinkhorn-Knopp algorithm (standard + log-domain). Hot loop uses faer SIMD mat-vec.
- `src/cost.rs` — Cost matrix computation. Rayon-parallel for N >= 100.
- `src/exact_1d.rs` — 1D exact OT via sorting.
- `src/divergence.rs` — Sinkhorn divergence (debiased).
- `src/sliced.rs` — Sliced Wasserstein via random projections. Scales to N=100k+.
- `src/unbalanced.rs` — Unbalanced OT with KL penalty.
- `src/barycenter.rs` — Fixed-support Wasserstein barycenters via Bregman projection.
- `src/error.rs` — Error types. All errors propagate as Python exceptions via PyO3.
- `src/lib.rs` — PyO3 bindings. All computation releases the GIL.

## Conventions

- Matrices are flat `Vec<f64>` in row-major order (converted to faer column-major at boundaries).
- All public Rust functions return `Result<T, KantorovichError>`.
- Python API uses `PyReadonlyArrayDyn<f64>` for input, `ArrayD` + `into_pyarray` for output.
- Tests go in `#[cfg(test)] mod tests` at the bottom of each file.
