/// Exact 1D optimal transport via sorting.
///
/// For 1D distributions, the optimal transport plan is given by the
/// monotone rearrangement: sort both distributions and match them
/// in order. This gives an O(n log n) exact solution.

/// Compute the Wasserstein-p distance between two 1D distributions.
///
/// # Arguments
/// * `x_a` - Support points of distribution a (length n)
/// * `a` - Weights of distribution a (length n, sums to 1)
/// * `x_b` - Support points of distribution b (length m)
/// * `b` - Weights of distribution b (length m, sums to 1)
/// * `p` - Order of the Wasserstein distance (typically 1 or 2)
///
/// # Returns
/// W_p(a, b) = (sum_k |x_{sigma(k)} - y_{tau(k)}|^p * w_k)^{1/p}
pub fn wasserstein_1d(
    x_a: &[f64],
    a: &[f64],
    x_b: &[f64],
    b: &[f64],
    p: f64,
) -> Result<f64, crate::error::KantorovichError> {
    crate::error::validate_probability_vector(a, "a")?;
    crate::error::validate_probability_vector(b, "b")?;

    if x_a.len() != a.len() {
        return Err(crate::error::KantorovichError::InvalidInput {
            message: format!(
                "x_a length ({}) != a length ({})",
                x_a.len(),
                a.len()
            ),
        });
    }
    if x_b.len() != b.len() {
        return Err(crate::error::KantorovichError::InvalidInput {
            message: format!(
                "x_b length ({}) != b length ({})",
                x_b.len(),
                b.len()
            ),
        });
    }

    let n = a.len();
    let m = b.len();

    // Sort both distributions by support points
    let mut sorted_a: Vec<(f64, f64)> = x_a.iter().copied().zip(a.iter().copied()).collect();
    sorted_a.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut sorted_b: Vec<(f64, f64)> = x_b.iter().copied().zip(b.iter().copied()).collect();
    sorted_b.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute Wasserstein distance via the quantile coupling
    // Walk through both sorted distributions simultaneously
    let mut cost = 0.0;
    let mut i = 0; // index into sorted_a
    let mut j = 0; // index into sorted_b
    let mut weight_a = sorted_a[0].1; // remaining weight at current a position
    let mut weight_b = sorted_b[0].1; // remaining weight at current b position

    while i < n && j < m {
        let transport_weight = weight_a.min(weight_b);
        let dist = (sorted_a[i].0 - sorted_b[j].0).abs();

        cost += transport_weight * dist.powf(p);

        weight_a -= transport_weight;
        weight_b -= transport_weight;

        if weight_a < 1e-15 {
            i += 1;
            if i < n {
                weight_a = sorted_a[i].1;
            }
        }
        if weight_b < 1e-15 {
            j += 1;
            if j < m {
                weight_b = sorted_b[j].1;
            }
        }
    }

    Ok(cost.powf(1.0 / p))
}

/// Compute the earth mover's distance (Wasserstein-1) between two 1D distributions.
pub fn emd_1d(
    x_a: &[f64],
    a: &[f64],
    x_b: &[f64],
    b: &[f64],
) -> Result<f64, crate::error::KantorovichError> {
    wasserstein_1d(x_a, a, x_b, b, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_1d_identical() {
        let x = vec![1.0, 2.0, 3.0];
        let a = vec![1.0 / 3.0; 3];
        let result = emd_1d(&x, &a, &x, &a).unwrap();
        assert!(result < 1e-10, "EMD of identical distributions should be 0, got {result}");
    }

    #[test]
    fn test_emd_1d_dirac() {
        // Two Dirac deltas at 0 and 1
        let x_a = vec![0.0];
        let a = vec![1.0];
        let x_b = vec![1.0];
        let b = vec![1.0];
        let result = emd_1d(&x_a, &a, &x_b, &b).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "EMD should be 1.0, got {result}");
    }

    #[test]
    fn test_emd_1d_shift() {
        // Uniform on {0, 1, 2} vs uniform on {1, 2, 3} -> EMD = 1
        let x_a = vec![0.0, 1.0, 2.0];
        let a = vec![1.0 / 3.0; 3];
        let x_b = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 / 3.0; 3];
        let result = emd_1d(&x_a, &a, &x_b, &b).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "EMD should be 1.0, got {result}");
    }

    #[test]
    fn test_wasserstein_2_diracs() {
        let x_a = vec![0.0];
        let a = vec![1.0];
        let x_b = vec![3.0];
        let b = vec![1.0];
        let result = wasserstein_1d(&x_a, &a, &x_b, &b, 2.0).unwrap();
        assert!((result - 3.0).abs() < 1e-10, "W2 should be 3.0, got {result}");
    }

    #[test]
    fn test_emd_1d_asymmetric() {
        // a = delta at 0, b = 0.5 at 1 + 0.5 at 3
        // Optimal: send all mass from 0 to closest available
        // EMD = 0.5 * |0-1| + 0.5 * |0-3| = 0.5 + 1.5 = 2.0
        let x_a = vec![0.0];
        let a = vec![1.0];
        let x_b = vec![1.0, 3.0];
        let b = vec![0.5, 0.5];
        let result = emd_1d(&x_a, &a, &x_b, &b).unwrap();
        assert!((result - 2.0).abs() < 1e-10, "EMD should be 2.0, got {result}");
    }
}
