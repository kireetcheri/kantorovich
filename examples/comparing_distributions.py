"""
Comparing probability distributions with kantorovich.

This example shows how to use optimal transport to measure the distance
between distributions — a fundamental operation in ML model evaluation,
statistical testing, and scientific computing.

Use case: You've trained a generative model and want to know how close
its output distribution is to the real data distribution. OT gives you
a geometrically meaningful answer (unlike KL divergence, which ignores
the metric structure of the space).
"""

import numpy as np
import kantorovich as kt

np.random.seed(42)


# --- Example 1: Are these two samples from the same distribution? ---

print("=" * 60)
print("Example 1: Two-sample testing with Sinkhorn divergence")
print("=" * 60)

# Ground truth: two samples from the same Gaussian
n = 500
x_same = np.random.randn(n, 3)
y_same = np.random.randn(n, 3)

# And one sample from a shifted Gaussian
y_shifted = np.random.randn(n, 3) + 0.5  # shifted by 0.5 in each dim

a = np.ones(n) / n
b = np.ones(n) / n

# Sinkhorn divergence: 0 means identical, larger means more different
# (Unlike raw Sinkhorn cost, the divergence is debiased: SD(a,a) = 0)
M_same = kt.cost_matrix(x_same, y_same)
M_xx = kt.cost_matrix(x_same, x_same)
M_yy_same = kt.cost_matrix(y_same, y_same)

sd_same = kt.sinkhorn_divergence_solve(a, b, M_same, M_xx, M_yy_same, reg=1.0)

M_shift = kt.cost_matrix(x_same, y_shifted)
M_yy_shift = kt.cost_matrix(y_shifted, y_shifted)

sd_shift = kt.sinkhorn_divergence_solve(a, b, M_shift, M_xx, M_yy_shift, reg=1.0)

print(f"Same distribution:    SD = {sd_same['divergence']:.6f}")
print(f"Shifted by 0.5:       SD = {sd_shift['divergence']:.6f}")
print(f"Ratio: {sd_shift['divergence'] / max(sd_same['divergence'], 1e-10):.1f}x larger")
print()


# --- Example 2: Sliced Wasserstein for large-scale comparison ---

print("=" * 60)
print("Example 2: Large-scale comparison with Sliced Wasserstein")
print("=" * 60)

# When N is large, computing the full NxN cost matrix is too expensive.
# Sliced Wasserstein projects onto random 1D directions and computes
# exact 1D OT — no NxN matrix needed.

import time

n_large = 50_000
d = 10

# Real data: mixture of two Gaussians
real_data = np.vstack([
    np.random.randn(n_large // 2, d) - 2,
    np.random.randn(n_large // 2, d) + 2,
])
np.random.shuffle(real_data)

# Good model: captures the bimodal structure
good_model = np.vstack([
    np.random.randn(n_large // 2, d) - 2,
    np.random.randn(n_large // 2, d) + 2,
])
np.random.shuffle(good_model)

# Bad model: single Gaussian (mode collapse)
bad_model = np.random.randn(n_large, d) * 2.5

weights = np.ones(n_large) / n_large

start = time.perf_counter()
sw_good = kt.sliced_wasserstein_solve(
    real_data, good_model, weights, weights,
    n_projections=100, p=2.0, seed=42
)
t_good = (time.perf_counter() - start) * 1000

start = time.perf_counter()
sw_bad = kt.sliced_wasserstein_solve(
    real_data, bad_model, weights, weights,
    n_projections=100, p=2.0, seed=42
)
t_bad = (time.perf_counter() - start) * 1000

print(f"Good model (bimodal): SW2 = {sw_good:.4f}  ({t_good:.0f}ms)")
print(f"Bad model (collapse): SW2 = {sw_bad:.4f}  ({t_bad:.0f}ms)")
print(f"The bad model is {sw_bad/sw_good:.1f}x further from real data.")
print(f"Computed on {n_large:,} points in {d}D — no NxN matrix needed.")
print()


# --- Example 3: Wasserstein barycenter ---

print("=" * 60)
print("Example 3: Wasserstein barycenter (distribution averaging)")
print("=" * 60)

# Given several distributions on a shared support, find their
# "average" in the Wasserstein sense. Unlike arithmetic averaging,
# this respects the geometry of the space.

support = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
M = kt.cost_matrix(support, support)

# Three distributions with mass at different locations
d1 = np.array([0.8, 0.15, 0.05, 0.0, 0.0])   # mass at left
d2 = np.array([0.0, 0.0, 0.05, 0.15, 0.8])   # mass at right
d3 = np.array([0.05, 0.1, 0.7, 0.1, 0.05])   # mass at center

# Equal-weight barycenter
result = kt.barycenter_solve(
    [d1, d2, d3], M,
    weights=np.array([1/3, 1/3, 1/3]),
    reg=0.1
)

bary = result["barycenter"]
print("Input distributions:")
print(f"  Left-heavy:   {np.array2string(d1, precision=2)}")
print(f"  Right-heavy:  {np.array2string(d2, precision=2)}")
print(f"  Center-heavy: {np.array2string(d3, precision=2)}")
print(f"Barycenter:     {np.array2string(bary, precision=2)}")
print(f"  (mass is spread across the support, respecting geometry)")
print()


# --- Example 4: Unbalanced OT ---

print("=" * 60)
print("Example 4: Unbalanced OT (handling outliers / mass mismatch)")
print("=" * 60)

# Standard OT requires both distributions to have the same total mass.
# Unbalanced OT relaxes this — useful when one distribution has outliers
# or when you're comparing datasets of different sizes.

# Scenario: comparing a clean signal to a noisy measurement
clean = np.array([0.3, 0.4, 0.3, 0.0])    # total mass = 1.0
noisy = np.array([0.2, 0.3, 0.2, 0.5])    # has extra mass (outlier at position 4)

support_4 = np.array([[0.0], [1.0], [2.0], [10.0]])  # position 4 is far away
M4 = kt.cost_matrix(support_4, support_4)

# Balanced OT would force all mass to be transported (including the outlier)
balanced = kt.sinkhorn_solve(
    clean / clean.sum(), noisy / noisy.sum(),
    M4, reg=0.5
)

# Unbalanced OT can "destroy" the outlier mass instead of transporting it
unbalanced = kt.sinkhorn_unbalanced_solve(
    clean, noisy, M4, reg=0.5, tau=0.5
)

print(f"Balanced OT cost:    {balanced['cost']:.4f}  (forced to transport outlier)")
print(f"Unbalanced OT cost:  {unbalanced['cost']:.4f}  (outlier mass destroyed)")
print(f"Unbalanced plan total mass: {unbalanced['plan'].sum():.3f} (< 1.0)")
print()

print("=" * 60)
print("All examples complete.")
print("=" * 60)
