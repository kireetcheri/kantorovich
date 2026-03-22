# Sinkhorn Divergence: Mathematical Derivation

## The Bias Problem

The entropic OT cost S_reg(a, b) is **not** a proper divergence because:

```
S_reg(a, a) > 0   for all reg > 0
```

Entropic regularization adds "blur" — even when comparing a distribution to itself, there's a positive cost due to the entropy term spreading mass.

This means S_reg is biased: it systematically overestimates the true transport cost, and the bias depends on reg in a non-trivial way.

## The Fix: Debiasing

Feydy et al. (2019) defined the Sinkhorn divergence:

```
SD_reg(a, b) = S_reg(a, b) - (1/2) S_reg(a, a) - (1/2) S_reg(b, b)
```

This subtracts the self-transport bias from both distributions.

## Properties

1. **Non-negative**: SD_reg(a, b) >= 0
2. **Identity**: SD_reg(a, a) = 0 exactly (by construction)
3. **Metrizes weak convergence**: SD_reg(a_n, b) -> 0 iff a_n -> b weakly
4. **Interpolation**: As reg -> 0, SD_reg -> W_p^p (Wasserstein distance). As reg -> inf, SD_reg -> (1/2)||a - b||^2 (MMD with Gaussian kernel).
5. **Differentiable**: Unlike Wasserstein distance, SD_reg is smooth (useful for optimization, e.g., training Wasserstein GANs)

## Why This Matters

In practice, if you're using OT to compare distributions (e.g., evaluating generative models), the raw Sinkhorn cost gives you a number that depends on both the actual distance AND the regularization parameter in a confounded way. The Sinkhorn divergence separates these: it gives you a clean measure of distance that you can meaningfully compare across different reg values.

## Computation

Three independent Sinkhorn solves:
1. S_reg(a, b) with cost matrix C_ab
2. S_reg(a, a) with cost matrix C_aa
3. S_reg(b, b) with cost matrix C_bb

Then combine: SD = S_ab - 0.5 * S_aa - 0.5 * S_bb

The three solves are independent and could be parallelized (future optimization).

## References

1. Feydy, J. et al. (2019). "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences." AISTATS.
2. Genevay, A. et al. (2018). "Learning Generative Models with Sinkhorn Divergences." AISTATS.
