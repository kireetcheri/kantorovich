/// Unbalanced optimal transport with KL divergence penalty.
///
/// Standard OT requires marginal constraints P1 = a and P^T 1 = b exactly.
/// Unbalanced OT relaxes these constraints with KL divergence penalties:
///
///   min <P, C> - reg * H(P) + tau * KL(P1 || a) + tau * KL(P^T 1 || b)
///
/// This is useful when:
/// - Source and target have different total mass
/// - There are outliers in one distribution
/// - Exact marginal matching is too strict
///
/// Solved via a modified Sinkhorn iteration with KL proximal step.
///
/// Reference: Chizat et al., "Scaling Algorithms for Unbalanced Transport
/// Problems", Mathematics of Computation, 2018.
use crate::error::KantorovichError;
use faer::Mat;

/// Result of unbalanced OT computation.
#[derive(Debug, Clone)]
pub struct UnbalancedResult {
    /// Transport plan P (flat row-major, n x m)
    pub plan: Vec<f64>,
    /// Transport cost <P, C>
    pub cost: f64,
    /// Number of iterations used
    pub iterations: usize,
    pub n_rows: usize,
    pub n_cols: usize,
}

/// Sinkhorn algorithm for unbalanced OT with KL penalty.
///
/// # Arguments
/// * `a` - Source distribution (length n, non-negative, need not sum to 1)
/// * `b` - Target distribution (length m, non-negative, need not sum to 1)
/// * `cost` - Cost matrix (flat row-major, n x m)
/// * `reg` - Entropic regularization parameter
/// * `tau` - KL divergence penalty weight (larger = closer to balanced OT)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn sinkhorn_unbalanced(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    reg: f64,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> Result<UnbalancedResult, KantorovichError> {
    let n = a.len();
    let m = b.len();

    // Validate inputs (relaxed: don't require sum to 1)
    if a.is_empty() || b.is_empty() {
        return Err(KantorovichError::InvalidInput {
            message: "Distributions must not be empty".to_string(),
        });
    }
    for (i, &val) in a.iter().enumerate() {
        if val < 0.0 || val.is_nan() {
            return Err(KantorovichError::InvalidInput {
                message: format!("a[{i}] = {val} is invalid (must be non-negative)"),
            });
        }
    }
    for (i, &val) in b.iter().enumerate() {
        if val < 0.0 || val.is_nan() {
            return Err(KantorovichError::InvalidInput {
                message: format!("b[{i}] = {val} is invalid (must be non-negative)"),
            });
        }
    }
    crate::error::validate_cost_matrix(cost, n, m)?;

    if reg <= 0.0 {
        return Err(KantorovichError::InvalidInput {
            message: format!("Regularization must be positive, got {reg}"),
        });
    }
    if tau <= 0.0 {
        return Err(KantorovichError::InvalidInput {
            message: format!("KL penalty tau must be positive, got {tau}"),
        });
    }

    // Proximal parameter: fi = tau / (tau + reg)
    let fi = tau / (tau + reg);

    // Compute Gibbs kernel: K[i,j] = exp(-C[i,j] / reg)
    let k_data: Vec<f64> = cost.iter().map(|&c| (-c / reg).exp()).collect();
    let k = Mat::from_fn(n, m, |i, j| k_data[i * m + j]);

    let mut u = vec![1.0; n];
    let mut v = vec![1.0; m];

    let mut iterations = 0;
    let mut achieved_tol = f64::MAX;

    for iter in 0..max_iter {
        let u_prev = u.clone();

        // Kv = K @ v
        let v_col = faer::Col::from_fn(m, |j| v[j]);
        let kv = &k * &v_col;

        // u = (a / Kv)^fi  — the KL proximal step
        for i in 0..n {
            let kv_i = kv[i];
            if kv_i > 1e-300 {
                u[i] = (a[i] / kv_i).powf(fi);
            } else {
                u[i] = 0.0;
            }
        }

        // Ktu = K^T @ u
        let u_col = faer::Col::from_fn(n, |i| u[i]);
        let ktu = k.transpose() * &u_col;

        // v = (b / Ktu)^fi  — the KL proximal step
        for j in 0..m {
            let ktu_j = ktu[j];
            if ktu_j > 1e-300 {
                v[j] = (b[j] / ktu_j).powf(fi);
            } else {
                v[j] = 0.0;
            }
        }

        // Check convergence: ||u - u_prev||_inf / max(||u||_inf, 1)
        if iter % 10 == 0 {
            let u_max = u.iter().fold(1.0_f64, |acc, &x| acc.max(x.abs()));
            let max_change = u
                .iter()
                .zip(u_prev.iter())
                .fold(0.0_f64, |acc, (&ui, &ui_prev)| {
                    acc.max((ui - ui_prev).abs())
                });

            achieved_tol = max_change / u_max;
            if achieved_tol < tol {
                iterations = iter + 1;
                break;
            }
        }

        iterations = iter + 1;
    }

    if achieved_tol >= tol && iterations == max_iter {
        return Err(KantorovichError::ConvergenceError {
            iterations,
            tolerance: tol,
            achieved: achieved_tol,
        });
    }

    // Compute transport plan: P[i,j] = u[i] * K[i,j] * v[j]
    let mut plan = vec![0.0; n * m];
    let mut transport_cost = 0.0;
    for i in 0..n {
        for j in 0..m {
            let p_ij = u[i] * k_data[i * m + j] * v[j];
            plan[i * m + j] = p_ij;
            transport_cost += p_ij * cost[i * m + j];
        }
    }

    Ok(UnbalancedResult {
        plan,
        cost: transport_cost,
        iterations,
        n_rows: n,
        n_cols: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unbalanced_balanced_case() {
        // When tau is very large, unbalanced OT should approximate balanced OT
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        let result = sinkhorn_unbalanced(&a, &b, &cost, 0.1, 100.0, 2000, 1e-4).unwrap();

        // Plan should be approximately diagonal
        assert!(result.plan[0] > 0.3, "Expected diagonal transport");
        assert!(result.plan[3] > 0.3, "Expected diagonal transport");
    }

    #[test]
    fn test_unbalanced_different_mass() {
        // a has total mass 1, b has total mass 2
        // Unbalanced OT should handle this gracefully
        let a = vec![0.5, 0.5];
        let b = vec![1.0, 1.0];
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        let result = sinkhorn_unbalanced(&a, &b, &cost, 0.1, 1.0, 1000, 1e-6).unwrap();

        // Should produce a valid plan (non-negative)
        for &p in &result.plan {
            assert!(p >= 0.0, "Plan should be non-negative");
        }
        assert!(result.cost >= 0.0, "Cost should be non-negative");
    }

    #[test]
    fn test_unbalanced_validates_input() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![1.0, 2.0, 3.0, 4.0];

        // Negative reg
        assert!(sinkhorn_unbalanced(&a, &b, &cost, -1.0, 1.0, 100, 1e-8).is_err());
        // Negative tau
        assert!(sinkhorn_unbalanced(&a, &b, &cost, 0.1, -1.0, 100, 1e-8).is_err());
    }

    #[test]
    fn test_unbalanced_small_tau() {
        // Small tau = more unbalanced (marginal constraints relaxed)
        // Total transported mass should be less than min(sum(a), sum(b))
        let a = vec![0.5, 0.5];
        let b = vec![0.0, 1.0]; // all mass at point 1
        let cost = vec![0.0, 10.0, 10.0, 0.0]; // expensive cross-transport

        let result = sinkhorn_unbalanced(&a, &b, &cost, 0.1, 0.1, 1000, 1e-6).unwrap();

        let total_mass: f64 = result.plan.iter().sum();
        // With small tau and expensive transport, some mass should be "destroyed"
        assert!(
            total_mass < 0.95,
            "Small tau should allow mass destruction, total={total_mass}"
        );
    }
}
