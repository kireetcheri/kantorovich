# Wasserstein Barycenters: Mathematical Derivation

## Problem Statement

Given K probability distributions {a_1, ..., a_K} on a shared support of N points, with weights {w_1, ..., w_K} (summing to 1), find the barycenter:

```
b* = argmin_b  sum_{k=1}^{K}  w_k * S_reg(a_k, b)
```

where S_reg is the entropic OT cost.

The barycenter is the "average" distribution in the Wasserstein sense — it respects the geometry of the space, unlike arithmetic averaging.

## Why Not Just Average?

Arithmetic mean: `b_arith = sum_k w_k * a_k`

This ignores geometry. Example on support {0, 1, 2}:
- `a_1 = [1, 0, 0]` (mass at 0)
- `a_2 = [0, 0, 1]` (mass at 2)
- Arithmetic mean: `[0.5, 0, 0.5]` (mass split between 0 and 2)
- Wasserstein barycenter: `[0, 1, 0]` (mass at 1 — the geometric midpoint!)

The Wasserstein barycenter "knows" that position 1 is between 0 and 2.

## Algorithm: Iterative Bregman Projection

Following Cuturi & Doucet (2014), we alternate between:

1. **Sinkhorn-like updates** for each distribution k:
   - Compute `Kv_k = K @ v_k` (matrix-vector product)
   - Compute `u_k = a_k ./ Kv_k` (scaling to match a_k marginal)
   - Compute `K^T u_k` (transpose product for barycenter update)

2. **Barycenter update** via weighted geometric mean:
   ```
   b_j = exp( sum_k w_k * log(v_k_j * (K^T u_k)_j) )
   ```
   then normalize: `b = b / sum(b)`

3. **Dual variable update** for each k:
   ```
   v_k = b ./ (K^T u_k)
   ```

## Convergence

The algorithm converges when the barycenter stops changing:

```
max_j |b_j^{new} - b_j^{old}| < tol
```

Convergence is typically fast (10-50 iterations) for reasonable reg values.

## Fixed vs Free Support

- **Fixed support** (what kantorovich implements): The N support points are given; we only optimize the weights b. This is a convex problem with a unique solution.

- **Free support** (not implemented): Both the support points and weights are optimized. This is non-convex and much harder — a research-level problem.

## Complexity

Per iteration:
- K matrix-vector products K @ v_k: O(K * N^2)
- K transpose products K^T @ u_k: O(K * N^2)
- Barycenter update: O(K * N)

Total: O(T * K * N^2) where T is iterations to convergence.

## References

1. Cuturi, M. & Doucet, A. (2014). "Fast Computation of Wasserstein Barycenters." ICML.
2. Benamou, J.-D. et al. (2015). "Iterative Bregman Projections for Regularized Transportation Problems." SIAM J. Sci. Comp.
