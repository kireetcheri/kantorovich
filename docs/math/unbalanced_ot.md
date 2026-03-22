# Unbalanced Optimal Transport: Mathematical Derivation

## Motivation

Standard OT requires exact marginal constraints: P@1 = a and P^T@1 = b. This means both distributions must have the same total mass (typically normalized to 1).

In practice, this is often too rigid:
- Comparing datasets of different sizes
- Distributions with outliers (forcing transport to/from outliers is expensive and misleading)
- Partial matching (only some mass should be transported)

## Formulation

Unbalanced OT relaxes the marginal constraints using KL divergence penalties:

```
min_{P >= 0}  <P, C> - reg * H(P) + tau * KL(P@1 || a) + tau * KL(P^T@1 || b)
```

where:
- `KL(p || q) = sum_i p_i log(p_i / q_i) - p_i + q_i` is the KL divergence
- `tau > 0` controls how strictly marginals are enforced
- As `tau -> infinity`, this recovers balanced OT
- As `tau -> 0`, marginals are fully relaxed (mass can be freely created/destroyed)

## Modified Sinkhorn Iteration

The optimal solution still has the form P = diag(u) K diag(v), but the updates include a proximal step:

```
u^{k+1} = ( a ./ (K @ v^k) )^fi
v^{k+1} = ( b ./ (K^T @ u^{k+1}) )^fi
```

where `fi = tau / (tau + reg)` is the proximal parameter.

**Comparison to balanced Sinkhorn**:
- Balanced: `u = a ./ (K @ v)` — exponent is 1
- Unbalanced: `u = (a ./ (K @ v))^fi` — exponent is fi in (0, 1)

When `fi = 1` (tau = infinity), we recover the balanced algorithm. When `fi < 1`, the update is "softened" — it doesn't fully enforce the marginal constraint.

## Intuition

The proximal parameter `fi` controls the trade-off:
- `fi close to 1` (large tau): marginals are almost exactly satisfied
- `fi close to 0` (small tau): marginals are barely enforced, mass can be freely created/destroyed
- The total mass of the transport plan `sum(P)` will generally be less than `min(sum(a), sum(b))` — some mass is "destroyed" rather than transported when the transport cost is too high

This is particularly useful for:
- **Outlier robustness**: outliers far from the other distribution have their mass destroyed rather than transported at high cost
- **Partial matching**: only the "easy" mass is transported

## References

1. Chizat, L. et al. (2018). "Scaling Algorithms for Unbalanced Transport Problems." Mathematics of Computation.
2. Séjourné, T. et al. (2019). "Sinkhorn Divergences for Unbalanced Optimal Transport." arXiv:1910.12958.
