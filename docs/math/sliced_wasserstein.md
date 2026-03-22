# Sliced Wasserstein Distance: Mathematical Derivation

## Motivation

The Wasserstein distance between d-dimensional distributions requires solving an OT problem with an N x N cost matrix — O(N^2) memory and O(N^2 * T) compute for Sinkhorn.

For large N (50,000+), this is infeasible. The sliced Wasserstein distance avoids this entirely.

## Key Insight: 1D OT is Cheap

In 1D, the Wasserstein-p distance has a closed-form solution via the quantile function:

```
W_p(mu, nu) = ( integral_0^1 |F_mu^{-1}(t) - F_nu^{-1}(t)|^p dt )^{1/p}
```

For discrete distributions, this reduces to sorting + matching:

```
W_p(a, b) = ( sum_k |x_{sigma(k)} - y_{tau(k)}|^p * w_k )^{1/p}
```

where sigma, tau are the sorting permutations. Complexity: O(N log N).

## Slicing: Project to 1D

The sliced Wasserstein distance projects both distributions onto random 1D directions and averages the 1D Wasserstein distances:

```
SW_p(mu, nu) = ( E_{theta ~ S^{d-1}} [ W_p^p(theta#mu, theta#nu) ] )^{1/p}
```

where:
- `theta` is a unit vector sampled uniformly from the (d-1)-sphere
- `theta#mu` is the pushforward (projection) of `mu` onto the direction `theta`
- `S^{d-1}` is the unit sphere in R^d

## Monte Carlo Approximation

In practice, we approximate the expectation with L random projections:

```
SW_p(mu, nu) ~= ( (1/L) sum_{l=1}^{L} W_p^p(theta_l # mu, theta_l # nu) )^{1/p}
```

For each projection:
1. Sample `theta_l` uniformly on S^{d-1} (via normalizing a Gaussian vector)
2. Project: `x_proj = X @ theta_l`, `y_proj = Y @ theta_l`
3. Compute exact 1D Wasserstein on the projections

## Properties

- **Lower bound**: SW_p(mu, nu) <= W_p(mu, nu) (equality in 1D)
- **Metric**: SW_p is a proper metric on probability distributions
- **Metrizes weak convergence**: SW_p(mu_n, mu) -> 0 iff mu_n -> mu weakly
- **Differentiable**: Unlike W_p, SW_p is smooth almost everywhere (useful for gradient-based optimization)
- **Scalable**: O(L * N * log(N) * d) — linear in N (versus quadratic for dense OT)

## Sampling Directions

To sample uniformly on S^{d-1}:
1. Generate d independent standard normal samples: z_i ~ N(0, 1)
2. Normalize: theta = z / ||z||_2

This works because the multivariate standard normal distribution is rotationally invariant.

## Complexity

| Component | Cost |
|---|---|
| Sample L directions | O(Ld) |
| Project N points | O(LNd) |
| Sort projections | O(LN log N) |
| 1D Wasserstein | O(LN) |
| **Total** | **O(LN(d + log N))** |

No N x N matrix is ever computed. This is why sliced Wasserstein scales to N = 100,000+.

## References

1. Rabin, J. et al. (2011). "Wasserstein Barycenter and its Application to Texture Mixing." SSVM.
2. Bonneel, N. et al. (2015). "Sliced and Radon Wasserstein Barycenters of Measures." JMIV.
3. Kolouri, S. et al. (2019). "Generalized Sliced Wasserstein Distances." NeurIPS.
