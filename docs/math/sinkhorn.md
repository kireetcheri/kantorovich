# Sinkhorn-Knopp Algorithm: Mathematical Derivation

## The Optimal Transport Problem

Given source distribution `a` in R^n and target distribution `b` in R^m on a cost matrix `C` in R^(n x m), the discrete optimal transport problem is:

```
min_{P >= 0}  <P, C>  =  sum_{ij} P_{ij} C_{ij}

subject to:  P @ 1_m = a     (row marginals)
             P^T @ 1_n = b   (column marginals)
```

where `P` is the transport plan (coupling matrix) and `<.,.>` is the Frobenius inner product.

This is a linear program with `nm` variables and `n+m` equality constraints. Exact solvers (e.g., network simplex) have O(n^3 log n) complexity.

## Entropic Regularization

Cuturi (2013) introduced entropic regularization to make OT scalable:

```
min_{P >= 0}  <P, C> - reg * H(P)

subject to:  P @ 1 = a,  P^T @ 1 = b
```

where `H(P) = -sum_{ij} P_{ij} log(P_{ij})` is the entropy.

**Key insight**: The optimal solution has the form:

```
P* = diag(u) @ K @ diag(v)
```

where `K_{ij} = exp(-C_{ij} / reg)` is the Gibbs kernel, and `u`, `v` are positive scaling vectors satisfying the marginal constraints.

## Standard Domain Algorithm

Substituting `P = diag(u) K diag(v)` into the marginal constraints:

```
P @ 1 = a  =>  diag(u) K diag(v) @ 1 = a  =>  u .* (K @ v) = a
P^T @ 1 = b  =>  diag(v) K^T diag(u) @ 1 = b  =>  v .* (K^T @ u) = b
```

This gives the Sinkhorn fixed-point iteration:

```
u^{k+1} = a ./ (K @ v^k)
v^{k+1} = b ./ (K^T @ u^{k+1})
```

where `./` is element-wise division.

**Convergence**: For `reg > 0` and strictly positive `a, b`, this converges linearly to the unique solution. The rate depends on `reg` — smaller `reg` means slower convergence but better approximation to unregularized OT.

**Convergence criterion**: We check the max absolute marginal violation:

```
tol_achieved = max( ||P@1 - a||_inf,  ||P^T@1 - b||_inf )
```

where `P@1_i = u_i * (K @ v)_i` and `P^T@1_j = v_j * (K^T @ u)_j`.

## Log-Domain Algorithm

For small `reg`, the Gibbs kernel `K_{ij} = exp(-C_{ij}/reg)` underflows to zero, causing division by zero in the standard algorithm.

The log-domain formulation works with dual variables `f = reg * log(u)`, `g = reg * log(v)`:

```
f_i = reg * log(a_i) - reg * LSE_j( (-C_{ij} + g_j) / reg )
g_j = reg * log(b_j) - reg * LSE_i( (-C_{ij} + f_i) / reg )
```

where `LSE(x) = log(sum(exp(x)))` is the log-sum-exp function, computed in a numerically stable way:

```
LSE(x) = max(x) + log( sum( exp(x - max(x)) ) )
```

The transport plan is recovered as:

```
P_{ij} = exp( (f_i + g_j - C_{ij}) / reg )
```

## Computational Complexity

Each Sinkhorn iteration requires:
- One matrix-vector product `K @ v` (O(nm) flops)
- One transposed matrix-vector product `K^T @ u` (O(nm) flops)
- Two element-wise divisions (O(n) + O(m) flops)

Total per iteration: O(nm). With T iterations to convergence: O(Tnm).

kantorovich uses faer's SIMD-accelerated matrix-vector multiply for the O(nm) operations, which is the bottleneck for large n, m.

## References

1. Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS.
2. Peyré, G. & Cuturi, M. (2019). "Computational Optimal Transport." Foundations and Trends in Machine Learning.
3. Schmitzer, B. (2019). "Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems." SIAM Journal on Scientific Computing.
