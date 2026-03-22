/// Sliced Wasserstein distance — fast approximation via random 1D projections.
///
/// Instead of solving the full d-dimensional OT problem (which requires an NxN
/// cost matrix), project both distributions onto random 1D directions and
/// compute the exact 1D Wasserstein distance for each projection. The average
/// over many projections approximates the true Wasserstein distance.
///
/// Complexity: O(n_projections * N * log(N)) — no NxN cost matrix needed.
/// Scales to N=100,000+ easily.
///
/// Reference: Rabin et al., "Wasserstein Barycenter and its Application to
/// Texture Mixing", SSVM 2011.
use crate::error::KantorovichError;

/// Compute the sliced Wasserstein distance between two point clouds.
///
/// # Arguments
/// * `x` - First point cloud (flat, n points x d dimensions)
/// * `y` - Second point cloud (flat, m points x d dimensions)
/// * `a` - Weights for x (length n, sums to 1)
/// * `b` - Weights for y (length m, sums to 1)
/// * `dim` - Dimensionality of points
/// * `n_projections` - Number of random projections (more = more accurate)
/// * `p` - Order of Wasserstein distance (typically 1 or 2)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Sliced Wasserstein-p distance
pub fn sliced_wasserstein(
    x: &[f64],
    y: &[f64],
    a: &[f64],
    b: &[f64],
    dim: usize,
    n_projections: usize,
    p: f64,
    seed: u64,
) -> Result<f64, KantorovichError> {
    let n = a.len();
    let m = b.len();

    if x.len() != n * dim {
        return Err(KantorovichError::InvalidInput {
            message: format!("x has {} elements, expected {} (n={} * dim={})", x.len(), n * dim, n, dim),
        });
    }
    if y.len() != m * dim {
        return Err(KantorovichError::InvalidInput {
            message: format!("y has {} elements, expected {} (m={} * dim={})", y.len(), m * dim, m, dim),
        });
    }

    crate::error::validate_probability_vector(a, "a")?;
    crate::error::validate_probability_vector(b, "b")?;

    let mut total_cost = 0.0;

    // Simple xorshift64 PRNG for reproducibility without external dependency
    let mut rng_state = seed;

    for _ in 0..n_projections {
        // Generate random direction on unit sphere
        // Using Box-Muller transform with xorshift64
        let direction = random_unit_vector(dim, &mut rng_state);

        // Project points onto direction
        let proj_x: Vec<f64> = (0..n)
            .map(|i| {
                let xi = &x[i * dim..(i + 1) * dim];
                xi.iter().zip(direction.iter()).map(|(&x, &d)| x * d).sum()
            })
            .collect();

        let proj_y: Vec<f64> = (0..m)
            .map(|j| {
                let yj = &y[j * dim..(j + 1) * dim];
                yj.iter().zip(direction.iter()).map(|(&y, &d)| y * d).sum()
            })
            .collect();

        // Compute exact 1D Wasserstein distance on projections
        let w = crate::exact_1d::wasserstein_1d(&proj_x, a, &proj_y, b, p)?;
        total_cost += w.powf(p);
    }

    Ok((total_cost / n_projections as f64).powf(1.0 / p))
}

/// Generate a random unit vector in d dimensions using Box-Muller + xorshift64.
fn random_unit_vector(dim: usize, state: &mut u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(dim);

    for _ in 0..dim {
        // Generate standard normal using Box-Muller
        let u1 = xorshift64_uniform(state);
        let u2 = xorshift64_uniform(state);
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v.push(normal);
    }

    // Normalize to unit sphere
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    v
}

/// Xorshift64 PRNG — returns uniform random in (0, 1).
#[inline]
fn xorshift64_uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    // Convert to (0, 1) range
    (*state as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliced_identical() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]; // 3 points in 2D
        let a = vec![1.0 / 3.0; 3];

        let dist = sliced_wasserstein(&x, &x, &a, &a, 2, 100, 2.0, 42).unwrap();
        assert!(dist < 1e-6, "SW of identical point clouds should be ~0, got {dist}");
    }

    #[test]
    fn test_sliced_shifted() {
        // Two point clouds, one shifted by (10, 0)
        let x = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let y = vec![10.0, 0.0, 11.0, 0.0, 12.0, 0.0];
        let a = vec![1.0 / 3.0; 3];

        let dist = sliced_wasserstein(&x, &y, &a, &a, 2, 200, 2.0, 42).unwrap();
        // Should be close to 10 (the shift distance)
        // Sliced Wasserstein is a lower bound on true Wasserstein, and
        // with finite projections in 2D there's variance, so we use a loose bound
        assert!(
            dist > 5.0 && dist < 15.0,
            "SW should be roughly ~10 for shift of 10, got {dist}"
        );
    }

    #[test]
    fn test_sliced_1d_matches_exact() {
        // In 1D, sliced Wasserstein with many projections should match exact
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![3.0, 4.0, 5.0];
        let a = vec![1.0 / 3.0; 3];

        // In 1D, the only "direction" is +1 or -1, so sliced should be exact
        let sliced = sliced_wasserstein(&x, &y, &a, &a, 1, 100, 1.0, 42).unwrap();
        let exact = crate::exact_1d::emd_1d(&x, &a, &y, &a).unwrap();

        assert!(
            (sliced - exact).abs() < 0.1,
            "Sliced W1 ({sliced}) should approximate exact W1 ({exact}) in 1D"
        );
    }

    #[test]
    fn test_sliced_large_n() {
        // Test that it handles large N without blowing up memory
        let n = 10000;
        let dim = 5;
        // Deterministic "random" points
        let x: Vec<f64> = (0..n * dim).map(|i| (i as f64 * 0.001).sin()).collect();
        let y: Vec<f64> = (0..n * dim).map(|i| (i as f64 * 0.001 + 1.0).sin()).collect();
        let a = vec![1.0 / n as f64; n];

        let dist = sliced_wasserstein(&x, &y, &a, &a, dim, 50, 2.0, 42).unwrap();
        assert!(dist > 0.0, "Distance should be positive");
        assert!(dist.is_finite(), "Distance should be finite");
    }
}
