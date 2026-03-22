"""Type stubs for kantorovich — high-performance optimal transport in Rust."""

from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

class SinkhornResult(TypedDict):
    plan: NDArray[np.float64]
    cost: float
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    iterations: int

class DivergenceResult(TypedDict):
    divergence: float
    cost_ab: float
    cost_aa: float
    cost_bb: float

class UnbalancedResult(TypedDict):
    plan: NDArray[np.float64]
    cost: float
    iterations: int

class BarycenterResult(TypedDict):
    barycenter: NDArray[np.float64]
    iterations: int

def sinkhorn_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    m: NDArray[np.float64],
    reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
    method: str = "standard",
) -> SinkhornResult:
    """Solve entropic regularized OT via Sinkhorn-Knopp.

    Args:
        a: Source distribution (1D, sums to 1).
        b: Target distribution (1D, sums to 1).
        m: Cost matrix (2D, shape (len(a), len(b))).
        reg: Entropic regularization (larger = more regularized).
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance (max marginal violation).
        method: "standard" (fast) or "log" (stable for small reg).

    Returns:
        Dict with plan, cost, dual variables u/v, and iteration count.

    Raises:
        ValueError: On invalid input or convergence failure.
    """
    ...

def emd_1d(
    x_a: NDArray[np.float64],
    a: NDArray[np.float64],
    x_b: NDArray[np.float64],
    b: NDArray[np.float64],
    p: float = 1.0,
) -> float:
    """Exact 1D Wasserstein-p distance via sorting. O(n log n).

    Args:
        x_a: Support points of distribution a.
        a: Weights of distribution a (sums to 1).
        x_b: Support points of distribution b.
        b: Weights of distribution b (sums to 1).
        p: Order of the Wasserstein distance.

    Returns:
        Wasserstein-p distance.
    """
    ...

def cost_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    metric: str = "sqeuclidean",
) -> NDArray[np.float64]:
    """Compute pairwise cost matrix between two point clouds.

    Args:
        x: First point cloud (2D, shape (n, d)).
        y: Second point cloud (2D, shape (m, d)).
        metric: "sqeuclidean", "euclidean", or "cosine".

    Returns:
        Cost matrix (2D, shape (n, m)).
    """
    ...

def sinkhorn_divergence_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    cost_ab: NDArray[np.float64],
    cost_aa: NDArray[np.float64],
    cost_bb: NDArray[np.float64],
    reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> DivergenceResult:
    """Debiased Sinkhorn divergence: SD(a,b) = S(a,b) - 0.5*S(a,a) - 0.5*S(b,b).

    A proper divergence (zero iff a == b) that metrizes weak convergence.

    Args:
        a, b: Distributions (1D, sum to 1).
        cost_ab: Cost matrix between a and b supports.
        cost_aa: Cost matrix of a support against itself.
        cost_bb: Cost matrix of b support against itself.
        reg: Entropic regularization.
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance.

    Returns:
        Dict with divergence value and component costs.
    """
    ...

def sliced_wasserstein_solve(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    n_projections: int = 50,
    p: float = 2.0,
    seed: int = 42,
) -> float:
    """Sliced Wasserstein distance via random 1D projections.

    Scales to N=100,000+ without computing an NxN cost matrix.

    Args:
        x, y: Point clouds (2D, shape (n, d) and (m, d)).
        a, b: Distribution weights (1D, sum to 1).
        n_projections: Number of random 1D projections.
        p: Wasserstein order.
        seed: Random seed for reproducibility.

    Returns:
        Sliced Wasserstein-p distance.
    """
    ...

def sinkhorn_unbalanced_solve(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    m: NDArray[np.float64],
    reg: float = 0.1,
    tau: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> UnbalancedResult:
    """Unbalanced OT with KL divergence penalty.

    Handles distributions with different total mass.

    Args:
        a, b: Distributions (non-negative, need not sum to 1).
        m: Cost matrix (2D).
        reg: Entropic regularization.
        tau: KL penalty weight (larger = closer to balanced OT).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Dict with transport plan, cost, and iteration count.
    """
    ...

def barycenter_solve(
    distributions: list[NDArray[np.float64]],
    cost: NDArray[np.float64],
    weights: NDArray[np.float64],
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BarycenterResult:
    """Fixed-support Wasserstein barycenter via Bregman projection.

    Args:
        distributions: List of distributions on the same support.
        cost: Cost matrix on the support (2D, n_support x n_support).
        weights: Weights for each distribution (sums to 1).
        reg: Entropic regularization.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on barycenter change.

    Returns:
        Dict with barycenter distribution and iteration count.
    """
    ...
