/// Fixed-support Wasserstein barycenters via iterative Bregman projection.
///
/// Given K distributions {(a_k, C_k)} on a fixed support of N points,
/// find the barycenter distribution b* that minimizes:
///
///   b* = argmin_b sum_k w_k * S_reg(a_k, b)
///
/// where w_k are weights for each input distribution and S_reg is the
/// entropic OT cost.
///
/// The algorithm alternates Sinkhorn-like updates for each distribution,
/// projecting onto the barycenter constraint.
///
/// Reference: Cuturi & Doucet, "Fast Computation of Wasserstein Barycenters",
/// ICML 2014.
use crate::error::KantorovichError;
use faer::Mat;

/// Result of barycenter computation.
#[derive(Debug, Clone)]
pub struct BarycenterResult {
    /// The barycenter distribution (length n_support)
    pub barycenter: Vec<f64>,
    /// Number of iterations used
    pub iterations: usize,
}

/// Compute the fixed-support Wasserstein barycenter.
///
/// # Arguments
/// * `distributions` - Slice of K distributions, each of length n_support
/// * `cost` - Cost matrix on the support (flat row-major, n_support x n_support)
/// * `weights` - Weights for each distribution (length K, sums to 1)
/// * `reg` - Entropic regularization
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance on barycenter change
///
/// # Returns
/// The barycenter distribution on the fixed support
pub fn free_support_barycenter(
    distributions: &[&[f64]],
    cost: &[f64],
    weights: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Result<BarycenterResult, KantorovichError> {
    let k = distributions.len();
    if k == 0 {
        return Err(KantorovichError::InvalidInput {
            message: "Need at least one distribution".to_string(),
        });
    }

    let n = distributions[0].len();

    // Validate all distributions have same length
    for (i, dist) in distributions.iter().enumerate() {
        if dist.len() != n {
            return Err(KantorovichError::InvalidInput {
                message: format!(
                    "Distribution {i} has length {}, expected {n}",
                    dist.len()
                ),
            });
        }
    }

    // Validate weights
    if weights.len() != k {
        return Err(KantorovichError::InvalidInput {
            message: format!("weights length {} != number of distributions {k}", weights.len()),
        });
    }
    let w_sum: f64 = weights.iter().sum();
    if (w_sum - 1.0).abs() > 1e-6 {
        return Err(KantorovichError::InvalidInput {
            message: format!("weights sum to {w_sum}, expected ~1.0"),
        });
    }

    crate::error::validate_cost_matrix(cost, n, n)?;

    if reg <= 0.0 {
        return Err(KantorovichError::InvalidInput {
            message: format!("Regularization must be positive, got {reg}"),
        });
    }

    // Compute Gibbs kernel: K[i,j] = exp(-C[i,j] / reg)
    let kernel = Mat::from_fn(n, n, |i, j| (-cost[i * n + j] / reg).exp());

    // Initialize dual variables v_k for each distribution
    let mut v: Vec<Vec<f64>> = vec![vec![1.0; n]; k];

    // Initialize barycenter as uniform
    let mut bary = vec![1.0 / n as f64; n];

    let mut iterations = 0;

    for iter in 0..max_iter {
        let bary_prev = bary.clone();

        // Compute weighted geometric mean of K^T diag(a_k / Kv_k)
        // for each support point
        let mut log_bary = vec![0.0; n];

        for s in 0..k {
            let a_s = distributions[s];

            // Compute Kv_s = K @ v_s
            let v_col = faer::Col::from_fn(n, |j| v[s][j]);
            let kv = &kernel * &v_col;

            // u_s = a_s / Kv_s
            // Then K^T u_s gives the column marginal contribution
            let u_col = faer::Col::from_fn(n, |i| {
                if kv[i] > 1e-300 {
                    a_s[i] / kv[i]
                } else {
                    0.0
                }
            });
            let ktu = kernel.transpose() * &u_col;

            // Accumulate weighted log for geometric mean
            for j in 0..n {
                let val = ktu[j] * v[s][j];
                if val > 1e-300 {
                    log_bary[j] += weights[s] * val.ln();
                } else {
                    log_bary[j] += weights[s] * (-300.0_f64).ln();
                }
            }
        }

        // Barycenter = exp(weighted log mean), then normalize
        for j in 0..n {
            bary[j] = log_bary[j].exp();
        }
        let bary_sum: f64 = bary.iter().sum();
        if bary_sum > 0.0 {
            for b in bary.iter_mut() {
                *b /= bary_sum;
            }
        }

        // Update v_k for each distribution: v_k = bary / (K^T u_k)
        for s in 0..k {
            let a_s = distributions[s];

            let v_col = faer::Col::from_fn(n, |j| v[s][j]);
            let kv = &kernel * &v_col;

            let u_col = faer::Col::from_fn(n, |i| {
                if kv[i] > 1e-300 {
                    a_s[i] / kv[i]
                } else {
                    0.0
                }
            });
            let ktu = kernel.transpose() * &u_col;

            for j in 0..n {
                if ktu[j] > 1e-300 {
                    v[s][j] = bary[j] / ktu[j];
                }
            }
        }

        // Check convergence
        let max_change: f64 = bary
            .iter()
            .zip(bary_prev.iter())
            .map(|(&b, &bp)| (b - bp).abs())
            .fold(0.0_f64, f64::max);

        iterations = iter + 1;
        if max_change < tol {
            break;
        }
    }

    Ok(BarycenterResult {
        barycenter: bary,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::sqeuclidean_cost_matrix;

    #[test]
    fn test_barycenter_single_distribution() {
        // Barycenter of a single distribution should be itself
        let a = vec![0.2, 0.3, 0.5];
        let x = vec![0.0, 1.0, 2.0];
        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result =
            free_support_barycenter(&[&a], &cost, &[1.0], 0.1, 100, 1e-6).unwrap();

        for (i, (&bi, &ai)) in result.barycenter.iter().zip(a.iter()).enumerate() {
            assert!(
                (bi - ai).abs() < 0.05,
                "Barycenter[{i}]={bi} should be close to a[{i}]={ai}"
            );
        }
    }

    #[test]
    fn test_barycenter_two_equal_weights() {
        // Barycenter of two uniform distributions should be uniform
        let a1 = vec![1.0 / 3.0; 3];
        let a2 = vec![1.0 / 3.0; 3];
        let x = vec![0.0, 1.0, 2.0];
        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result = free_support_barycenter(
            &[&a1, &a2],
            &cost,
            &[0.5, 0.5],
            0.1,
            100,
            1e-6,
        )
        .unwrap();

        for (i, &bi) in result.barycenter.iter().enumerate() {
            assert!(
                (bi - 1.0 / 3.0).abs() < 0.05,
                "Barycenter[{i}]={bi} should be ~1/3"
            );
        }
    }

    #[test]
    fn test_barycenter_sums_to_one() {
        let a1 = vec![0.8, 0.1, 0.1];
        let a2 = vec![0.1, 0.1, 0.8];
        let x = vec![0.0, 1.0, 2.0];
        let cost = sqeuclidean_cost_matrix(&x, &x, 1);

        let result = free_support_barycenter(
            &[&a1, &a2],
            &cost,
            &[0.5, 0.5],
            0.1,
            100,
            1e-6,
        )
        .unwrap();

        let sum: f64 = result.barycenter.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Barycenter should sum to 1, got {sum}"
        );
    }
}
