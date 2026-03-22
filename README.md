# kantorovich

High-performance optimal transport library in Rust with Python bindings.

Named after [Leonid Kantorovich](https://en.wikipedia.org/wiki/Leonid_Kantorovich), who formalized optimal transport theory and won the 1975 Nobel Prize in Economics.

## Why kantorovich?

- **Fast**: 2-4x faster than [POT](https://pythonot.github.io/) at typical problem sizes (N=100-1000), competitive at N=5000+
- **Pure Rust core**: SIMD-optimized via [faer](https://github.com/sarah-ek/faer-rs), parallel cost matrices via [rayon](https://github.com/rayon-rs/rayon)
- **Zero-copy Python bindings**: NumPy arrays in, NumPy arrays out. GIL released during all computation
- **Mathematically rigorous**: Both standard and log-domain Sinkhorn, proper convergence criteria, NaN/Inf detection

## Installation

```bash
pip install kantorovich
```

For development:

```bash
git clone <repo>
cd kantorovich
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install maturin numpy
maturin develop --release
```

## Quick Start

```python
import numpy as np
import kantorovich as kt

# Two point clouds in 5D
x = np.random.randn(500, 5)
y = np.random.randn(500, 5)
a, b = np.ones(500) / 500, np.ones(500) / 500

# Compute cost matrix and solve OT
M = kt.cost_matrix(x, y, metric="sqeuclidean")
result = kt.sinkhorn_solve(a, b, M, reg=0.1)
print(f"Transport cost: {result['cost']:.4f} ({result['iterations']} iterations)")
```

## API Reference

### Core OT

#### `sinkhorn_solve(a, b, M, reg=0.1, max_iter=1000, tol=1e-8, method="standard")`

Entropic regularized OT via Sinkhorn-Knopp.

- **method**: `"standard"` (fast) or `"log"` (stable for small `reg`)
- **Returns**: `{"plan", "cost", "u", "v", "iterations"}`

```python
result = kt.sinkhorn_solve(a, b, M, reg=0.01, method="log")
```

#### `emd_1d(x_a, a, x_b, b, p=1.0)`

Exact 1D Wasserstein-p distance via sorting. O(n log n).

```python
dist = kt.emd_1d(x_a, weights_a, x_b, weights_b, p=2.0)
```

### Distances & Divergences

#### `sinkhorn_divergence_solve(a, b, cost_ab, cost_aa, cost_bb, reg=0.1, ...)`

Debiased Sinkhorn divergence: `SD(a,b) = S(a,b) - 0.5*S(a,a) - 0.5*S(b,b)`.

Unlike raw Sinkhorn cost, this is a proper divergence (zero iff a == b) and metrizes weak convergence.

```python
M_ab = kt.cost_matrix(x, y)
M_aa = kt.cost_matrix(x, x)
M_bb = kt.cost_matrix(y, y)
result = kt.sinkhorn_divergence_solve(a, b, M_ab, M_aa, M_bb, reg=0.5)
print(result["divergence"])
```

#### `sliced_wasserstein_solve(x, y, a, b, n_projections=50, p=2.0, seed=42)`

Sliced Wasserstein distance via random 1D projections. **Scales to N=100,000+** — no NxN cost matrix needed.

```python
# 50,000 points in 10D — runs in ~350ms
sw = kt.sliced_wasserstein_solve(x, y, a, b, n_projections=100, p=2.0)
```

### Unbalanced & Barycenters

#### `sinkhorn_unbalanced_solve(a, b, M, reg=0.1, tau=1.0, ...)`

Unbalanced OT with KL divergence penalty. Handles distributions with different total mass.

- **tau**: KL penalty weight. Larger = closer to balanced OT.

```python
a = np.array([0.5, 0.5])       # total mass = 1
b = np.array([1.0, 1.0])       # total mass = 2
result = kt.sinkhorn_unbalanced_solve(a, b, M, reg=0.1, tau=1.0)
```

#### `barycenter_solve(distributions, cost, weights, reg=0.1, ...)`

Fixed-support Wasserstein barycenter via iterative Bregman projection.

```python
d1 = np.array([0.8, 0.1, 0.1])  # mass at left
d2 = np.array([0.1, 0.1, 0.8])  # mass at right
M = kt.cost_matrix(support, support)
result = kt.barycenter_solve([d1, d2], M, weights=np.array([0.5, 0.5]))
# result["barycenter"] has mass concentrated at center
```

### Utilities

#### `cost_matrix(x, y, metric="sqeuclidean")`

Pairwise cost matrix. Metrics: `"sqeuclidean"`, `"euclidean"`, `"cosine"`. Parallelized via rayon for N >= 100.

## Benchmarks

Sinkhorn algorithm, `reg=1.0`, median of 10 runs, release build with `RUSTFLAGS="-C target-cpu=native"`:

| N | kantorovich | POT | Speedup |
|---|---|---|---|
| 100 | 0.5ms | 1.9ms | **3.6x** |
| 200 | 1.8ms | 5.1ms | **2.8x** |
| 500 | 9.7ms | 12ms | **1.2x** |
| 1,000 | 39ms | 65ms | **1.7x** |
| 2,000 | 223ms | 240ms | **1.1x** |
| 5,000 | 1188ms | 1302ms | **1.1x** |

Sliced Wasserstein (no NxN matrix needed):

| N | d | Projections | Time |
|---|---|---|---|
| 1,000 | 10 | 100 | 10ms |
| 10,000 | 5 | 50 | 52ms |
| 50,000 | 5 | 50 | 316ms |
| 100,000 | 5 | 50 | 741ms |

1D exact OT:

| N | Time |
|---|---|
| 10,000 | 0.4ms |
| 100,000 | 4.1ms |
| 1,000,000 | 29ms |

## Architecture

```
src/
├── lib.rs          # PyO3 bindings, Python API surface
├── sinkhorn.rs     # Sinkhorn-Knopp (standard + log-domain, faer SIMD mat-vec)
├── exact_1d.rs     # 1D exact OT via sorting
├── cost.rs         # Cost matrices (rayon-parallel for N >= 100)
├── divergence.rs   # Sinkhorn divergence (debiased)
├── sliced.rs       # Sliced Wasserstein (random projections)
├── unbalanced.rs   # Unbalanced OT (KL penalty)
├── barycenter.rs   # Fixed-support Wasserstein barycenters
└── error.rs        # Error types (ConvergenceError, InvalidInput, NumericalError)
```

## Mathematical Background

**Optimal transport** finds the minimum-cost way to transform one probability distribution into another:

```
min  <P, C>  subject to  P >= 0,  P*1 = a,  P^T*1 = b
```

The **Sinkhorn algorithm** adds entropic regularization (`-reg * H(P)`) which makes the problem solvable via iterative matrix scaling. The key operation is alternating:
- `u = a / (K @ v)` where `K[i,j] = exp(-C[i,j] / reg)`
- `v = b / (K^T @ u)`

kantorovich implements this with faer's SIMD-accelerated matrix-vector products, achieving competitive performance with BLAS-backed implementations.

## License

MIT
