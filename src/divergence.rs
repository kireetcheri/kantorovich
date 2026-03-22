/// Sinkhorn divergence — a debiased version of the entropic OT cost.
///
/// The standard Sinkhorn cost S_reg(a, b) is biased: S_reg(a, a) != 0.
/// The Sinkhorn divergence corrects this:
///
///   SD_reg(a, b) = S_reg(a, b) - 0.5 * S_reg(a, a) - 0.5 * S_reg(b, b)
///
/// This gives a proper divergence that:
/// - Is non-negative
/// - Equals zero iff a == b
/// - Metrizes weak convergence (like Wasserstein distance)
/// - Is differentiable (unlike Wasserstein distance)
///
/// Reference: Feydy et al., "Interpolating between Optimal Transport and MMD
/// using Sinkhorn Divergences", AISTATS 2019.
use crate::error::KantorovichError;
use crate::sinkhorn;

/// Result of Sinkhorn divergence computation.
#[derive(Debug, Clone)]
pub struct DivergenceResult {
    /// The Sinkhorn divergence value
    pub divergence: f64,
    /// S_reg(a, b) — the cross term
    pub cost_ab: f64,
    /// S_reg(a, a) — the self-transport term for a
    pub cost_aa: f64,
    /// S_reg(b, b) — the self-transport term for b
    pub cost_bb: f64,
}

/// Compute the Sinkhorn divergence between distributions a and b.
///
/// SD_reg(a, b) = S_reg(a, b) - 0.5 * S_reg(a, a) - 0.5 * S_reg(b, b)
///
/// # Arguments
/// * `a` - Source distribution weights (length n, sums to 1)
/// * `b` - Target distribution weights (length m, sums to 1)
/// * `cost_ab` - Cost matrix between a and b (flat row-major, n x m)
/// * `cost_aa` - Cost matrix of a against itself (flat row-major, n x n)
/// * `cost_bb` - Cost matrix of b against itself (flat row-major, m x m)
/// * `reg` - Regularization parameter
/// * `max_iter` - Maximum Sinkhorn iterations
/// * `tol` - Convergence tolerance
pub fn sinkhorn_divergence(
    a: &[f64],
    b: &[f64],
    cost_ab: &[f64],
    cost_aa: &[f64],
    cost_bb: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Result<DivergenceResult, KantorovichError> {
    // S_reg(a, b)
    let result_ab = sinkhorn::sinkhorn(a, b, cost_ab, reg, max_iter, tol, false)?;

    // S_reg(a, a)
    let result_aa = sinkhorn::sinkhorn(a, a, cost_aa, reg, max_iter, tol, false)?;

    // S_reg(b, b)
    let result_bb = sinkhorn::sinkhorn(b, b, cost_bb, reg, max_iter, tol, false)?;

    let divergence = result_ab.cost - 0.5 * result_aa.cost - 0.5 * result_bb.cost;

    Ok(DivergenceResult {
        divergence: divergence.max(0.0), // Clamp to non-negative (numerical precision)
        cost_ab: result_ab.cost,
        cost_aa: result_aa.cost,
        cost_bb: result_bb.cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::sqeuclidean_cost_matrix;

    #[test]
    fn test_divergence_same_distribution() {
        // SD(a, a) should be ~0
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let x = vec![0.0, 1.0, 2.0, 3.0]; // 1D points
        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result = sinkhorn_divergence(&a, &a, &cost, &cost, &cost, 0.1, 1000, 1e-8).unwrap();

        assert!(
            result.divergence < 1e-6,
            "Divergence of identical distributions should be ~0, got {}",
            result.divergence
        );
    }

    #[test]
    fn test_divergence_different_distributions() {
        // SD(a, b) should be > 0 for different distributions
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let x = vec![0.0, 1.0, 2.0, 3.0];

        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result = sinkhorn_divergence(&a, &b, &cost, &cost, &cost, 0.1, 1000, 1e-8).unwrap();

        assert!(
            result.divergence > 0.1,
            "Divergence of different distributions should be > 0, got {}",
            result.divergence
        );
    }

    #[test]
    fn test_divergence_symmetry() {
        let a = vec![0.3, 0.7];
        let b = vec![0.6, 0.4];
        let x = vec![0.0, 1.0];

        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result_ab =
            sinkhorn_divergence(&a, &b, &cost, &cost, &cost, 0.5, 1000, 1e-6).unwrap();
        let result_ba =
            sinkhorn_divergence(&b, &a, &cost, &cost, &cost, 0.5, 1000, 1e-6).unwrap();

        assert!(
            (result_ab.divergence - result_ba.divergence).abs() < 1e-6,
            "Divergence should be symmetric: {} vs {}",
            result_ab.divergence,
            result_ba.divergence
        );
    }
}
