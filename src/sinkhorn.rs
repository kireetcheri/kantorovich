/// Sinkhorn-Knopp algorithm for entropic optimal transport.
///
/// Solves: min <P, C> - reg * H(P)
///         s.t. P1 = a, P^T 1 = b, P >= 0
///
/// where H(P) = -sum_{ij} P_{ij} log(P_{ij}) is the entropy.
///
/// Uses faer for SIMD-accelerated matrix-vector products in the hot loop.
use crate::error::KantorovichError;
use faer::{Col, Mat};

/// Result of Sinkhorn computation.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// Transport plan P (flat row-major, n x m)
    pub plan: Vec<f64>,
    /// Optimal transport cost <P, C>
    pub cost: f64,
    /// Dual variable u (length n)
    pub u: Vec<f64>,
    /// Dual variable v (length m)
    pub v: Vec<f64>,
    /// Number of iterations used
    pub iterations: usize,
    /// Dimensions of the transport plan
    pub n_rows: usize,
    pub n_cols: usize,
}

/// Build a faer Mat from a flat row-major slice.
#[inline]
fn mat_from_flat(data: &[f64], nrows: usize, ncols: usize) -> Mat<f64> {
    Mat::from_fn(nrows, ncols, |i, j| data[i * ncols + j])
}

/// Copy a slice into a faer Col (reusing existing allocation).
#[inline]
fn copy_to_col(col: &mut Col<f64>, data: &[f64]) {
    for (i, &val) in data.iter().enumerate() {
        col[i] = val;
    }
}

/// Sinkhorn-Knopp algorithm in standard (linear) domain.
///
/// Uses faer for SIMD-accelerated matrix-vector products with
/// pre-allocated buffers to eliminate per-iteration allocation.
pub fn sinkhorn_standard(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Result<SinkhornResult, KantorovichError> {
    let n = a.len();
    let m = b.len();

    // Compute Gibbs kernel: K[i,j] = exp(-C[i,j] / reg)
    let k_data: Vec<f64> = cost.iter().map(|&c| (-c / reg).exp()).collect();
    let k = mat_from_flat(&k_data, n, m);

    // Pre-allocate reusable Col buffers (eliminates per-iteration allocation)
    let mut u = vec![1.0; n];
    let mut v = vec![1.0; m];
    let mut v_col = Col::<f64>::zeros(m);
    let mut u_col = Col::<f64>::zeros(n);

    let mut iterations = 0;
    let mut achieved_tol = f64::MAX;

    for iter in 0..max_iter {
        // kv = K @ v (faer SIMD mat-vec)
        copy_to_col(&mut v_col, &v);
        let kv = &k * &v_col;

        // u = a / kv
        for i in 0..n {
            let kv_i = kv[i];
            if kv_i > 0.0 {
                u[i] = a[i] / kv_i;
            } else {
                return Err(KantorovichError::NumericalError {
                    message: format!("K*v is zero at row {i}, iteration {iter}. Try larger regularization."),
                });
            }
        }

        // ktu = K^T @ u (faer SIMD mat-vec)
        copy_to_col(&mut u_col, &u);
        let ktu = k.transpose() * &u_col;

        // v = b / ktu
        for j in 0..m {
            let ktu_j = ktu[j];
            if ktu_j > 0.0 {
                v[j] = b[j] / ktu_j;
            } else {
                return Err(KantorovichError::NumericalError {
                    message: format!("K^T*u is zero at col {j}, iteration {iter}. Try larger regularization."),
                });
            }
        }

        // Check convergence every 10 iterations
        if iter % 10 == 0 {
            copy_to_col(&mut v_col, &v);
            let kv = &k * &v_col;

            let mut max_violation = 0.0_f64;

            for i in 0..n {
                max_violation = max_violation.max((u[i] * kv[i] - a[i]).abs());
            }

            copy_to_col(&mut u_col, &u);
            let ktu = k.transpose() * &u_col;
            for j in 0..m {
                max_violation = max_violation.max((v[j] * ktu[j] - b[j]).abs());
            }

            achieved_tol = max_violation;
            if max_violation < tol {
                iterations = iter + 1;
                break;
            }
        }

        iterations = iter + 1;
    }

    if achieved_tol >= tol {
        return Err(KantorovichError::ConvergenceError {
            iterations,
            tolerance: tol,
            achieved: achieved_tol,
        });
    }

    // Compute transport plan P[i,j] = u[i] * K[i,j] * v[j] and cost
    let mut plan = vec![0.0; n * m];
    let mut transport_cost = 0.0;
    for i in 0..n {
        for j in 0..m {
            let p_ij = u[i] * k_data[i * m + j] * v[j];
            plan[i * m + j] = p_ij;
            transport_cost += p_ij * cost[i * m + j];
        }
    }

    Ok(SinkhornResult {
        plan,
        cost: transport_cost,
        u,
        v,
        iterations,
        n_rows: n,
        n_cols: m,
    })
}

/// Sinkhorn-Knopp algorithm in log domain.
///
/// Numerically stable for small regularization parameters.
/// Uses log-sum-exp to avoid underflow/overflow.
pub fn sinkhorn_log(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Result<SinkhornResult, KantorovichError> {
    let n = a.len();
    let m = b.len();

    // Precompute M/reg matrix for use in log-sum-exp
    // We'll use faer for the softmin operations
    let m_reg: Vec<f64> = cost.iter().map(|&c| c / reg).collect();
    let m_reg_mat = mat_from_flat(&m_reg, n, m);

    let log_a: Vec<f64> = a.iter().map(|&ai| ai.ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&bj| bj.ln()).collect();

    // Dual variables in log domain
    let mut f = vec![0.0; n];
    let mut g = vec![0.0; m];

    let mut iterations = 0;
    let mut achieved_tol = f64::MAX;

    for iter in 0..max_iter {
        // Update f: f_i = log(a_i) - logsumexp_j(-M[i,j]/reg + g_j/reg)
        // Which is: f_i/reg = log(a_i) - logsumexp_j(-M[i,j]/reg + g_j/reg)
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..m {
                let val = -m_reg_mat[(i, j)] + g[j] / reg;
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0;
            for j in 0..m {
                sum_exp += (-m_reg_mat[(i, j)] + g[j] / reg - max_val).exp();
            }

            f[i] = reg * (log_a[i] - max_val - sum_exp.ln());

            if f[i].is_nan() || f[i].is_infinite() {
                return Err(KantorovichError::NumericalError {
                    message: format!(
                        "NaN/Inf in dual variable f[{i}] at iteration {iter}. \
                         Try larger regularization (current: {reg:.2e})."
                    ),
                });
            }
        }

        // Update g: g_j = log(b_j) - logsumexp_i(-M[i,j]/reg + f_i/reg)
        for j in 0..m {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..n {
                let val = -m_reg_mat[(i, j)] + f[i] / reg;
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0;
            for i in 0..n {
                sum_exp += (-m_reg_mat[(i, j)] + f[i] / reg - max_val).exp();
            }

            g[j] = reg * (log_b[j] - max_val - sum_exp.ln());

            if g[j].is_nan() || g[j].is_infinite() {
                return Err(KantorovichError::NumericalError {
                    message: format!(
                        "NaN/Inf in dual variable g[{j}] at iteration {iter}. \
                         Try larger regularization (current: {reg:.2e})."
                    ),
                });
            }
        }

        // Check convergence every 10 iterations
        if iter % 10 == 0 {
            let mut max_violation = 0.0_f64;

            for i in 0..n {
                let mut row_sum = 0.0;
                for j in 0..m {
                    row_sum += ((f[i] + g[j] - cost[i * m + j]) / reg).exp();
                }
                max_violation = max_violation.max((row_sum - a[i]).abs());
            }

            for j in 0..m {
                let mut col_sum = 0.0;
                for i in 0..n {
                    col_sum += ((f[i] + g[j] - cost[i * m + j]) / reg).exp();
                }
                max_violation = max_violation.max((col_sum - b[j]).abs());
            }

            achieved_tol = max_violation;
            if max_violation < tol {
                iterations = iter + 1;
                break;
            }
        }

        iterations = iter + 1;
    }

    if achieved_tol >= tol {
        return Err(KantorovichError::ConvergenceError {
            iterations,
            tolerance: tol,
            achieved: achieved_tol,
        });
    }

    // Recover transport plan and cost
    let mut plan = vec![0.0; n * m];
    let mut transport_cost = 0.0;

    let u: Vec<f64> = f.iter().map(|&fi| (fi / reg).exp()).collect();
    let v: Vec<f64> = g.iter().map(|&gj| (gj / reg).exp()).collect();

    for i in 0..n {
        for j in 0..m {
            plan[i * m + j] = ((f[i] + g[j] - cost[i * m + j]) / reg).exp();
            transport_cost += plan[i * m + j] * cost[i * m + j];
        }
    }

    Ok(SinkhornResult {
        plan,
        cost: transport_cost,
        u,
        v,
        iterations,
        n_rows: n,
        n_cols: m,
    })
}

/// High-level Sinkhorn interface that dispatches to the appropriate method.
pub fn sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
    log_domain: bool,
) -> Result<SinkhornResult, KantorovichError> {
    crate::error::validate_probability_vector(a, "a")?;
    crate::error::validate_probability_vector(b, "b")?;
    crate::error::validate_cost_matrix(cost, a.len(), b.len())?;

    if reg <= 0.0 {
        return Err(KantorovichError::InvalidInput {
            message: format!("Regularization must be positive, got {reg}"),
        });
    }

    if log_domain {
        sinkhorn_log(a, b, cost, reg, max_iter, tol)
    } else {
        sinkhorn_standard(a, b, cost, reg, max_iter, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    #[test]
    fn test_sinkhorn_standard_uniform_2x2() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        let result = sinkhorn_standard(&a, &b, &cost, 0.1, 1000, 1e-8).unwrap();
        assert!(result.cost < 0.05, "Cost too high: {}", result.cost);
        assert!(result.plan[0] > 0.4);
        assert!(result.plan[3] > 0.4);
    }

    #[test]
    fn test_sinkhorn_log_uniform_2x2() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        let result = sinkhorn_log(&a, &b, &cost, 0.1, 1000, 1e-8).unwrap();
        assert!(result.cost < 0.05, "Cost too high: {}", result.cost);
        assert!(result.plan[0] > 0.4);
        assert!(result.plan[3] > 0.4);
    }

    #[test]
    fn test_sinkhorn_standard_and_log_agree() {
        let a = vec![0.3, 0.7];
        let b = vec![0.5, 0.5];
        let cost = vec![1.0, 2.0, 3.0, 0.5];
        let reg = 0.5;

        let std_result = sinkhorn_standard(&a, &b, &cost, reg, 1000, 1e-8).unwrap();
        let log_result = sinkhorn_log(&a, &b, &cost, reg, 1000, 1e-8).unwrap();

        assert!(
            (std_result.cost - log_result.cost).abs() < 1e-6,
            "Standard cost {} != Log cost {}",
            std_result.cost,
            log_result.cost
        );

        for (p1, p2) in std_result.plan.iter().zip(log_result.plan.iter()) {
            assert!((p1 - p2).abs() < 1e-6, "Plans differ: {} vs {}", p1, p2);
        }
    }

    #[test]
    fn test_sinkhorn_known_solution() {
        let a = uniform(2);
        let b = uniform(2);
        let cost = vec![0.0, 1.0, 1.0, 0.0];

        // High reg -> near uniform
        let high_reg = sinkhorn_standard(&a, &b, &cost, 10.0, 1000, 1e-8).unwrap();
        for &p in &high_reg.plan {
            assert!((p - 0.25).abs() < 0.05, "Expected near-uniform, got {}", p);
        }

        // Low reg -> near diagonal
        let low_reg = sinkhorn_log(&a, &b, &cost, 0.01, 5000, 1e-6).unwrap();
        assert!(low_reg.plan[0] > 0.45);
        assert!(low_reg.plan[3] > 0.45);
    }

    #[test]
    fn test_sinkhorn_validates_input() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let cost = vec![1.0, 2.0, 3.0, 4.0];

        let result = sinkhorn(&a, &b, &cost, -1.0, 100, 1e-8, false);
        assert!(result.is_err());

        let bad_cost = vec![1.0, 2.0];
        let result = sinkhorn(&a, &b, &bad_cost, 0.1, 100, 1e-8, false);
        assert!(result.is_err());

        let bad_a = vec![-0.5, 1.5];
        let result = sinkhorn(&bad_a, &b, &cost, 0.1, 100, 1e-8, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_sinkhorn_3x3() {
        let a = vec![1.0 / 3.0; 3];
        let b = vec![1.0 / 3.0; 3];
        let cost = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];

        let result = sinkhorn(&a, &b, &cost, 0.1, 1000, 1e-8, false).unwrap();

        let nn = 3;
        for i in 0..nn {
            let row_sum: f64 = (0..nn).map(|j| result.plan[i * nn + j]).sum();
            assert!((row_sum - a[i]).abs() < 1e-6, "Row {i} marginal: {row_sum}");
        }
        for j in 0..nn {
            let col_sum: f64 = (0..nn).map(|i| result.plan[i * nn + j]).sum();
            assert!((col_sum - b[j]).abs() < 1e-6, "Col {j} marginal: {col_sum}");
        }
    }
}
