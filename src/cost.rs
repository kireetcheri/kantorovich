/// Cost matrix computation for optimal transport.
///
/// All cost matrices are stored as flat Vec<f64> in row-major order.
/// Uses rayon for parallel computation on large matrices.
use rayon::prelude::*;

/// Threshold for switching to parallel computation.
const PARALLEL_THRESHOLD: usize = 100;

/// Compute squared Euclidean cost matrix between two point clouds.
///
/// Points are stored as flat arrays: x has n points of dimension d,
/// so x.len() == n * d. Similarly y has m points, y.len() == m * d.
///
/// Returns flat row-major matrix of shape (n, m) where C[i,j] = ||x_i - y_j||^2.
pub fn sqeuclidean_cost_matrix(x: &[f64], y: &[f64], dim: usize) -> Vec<f64> {
    let n = x.len() / dim;
    let m = y.len() / dim;

    if n >= PARALLEL_THRESHOLD {
        // Parallel: each row computed independently
        let cost: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let xi = &x[i * dim..(i + 1) * dim];
                (0..m)
                    .map(|j| {
                        let yj = &y[j * dim..(j + 1) * dim];
                        xi.iter()
                            .zip(yj.iter())
                            .map(|(&a, &b)| {
                                let d = a - b;
                                d * d
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        cost
    } else {
        // Sequential for small matrices
        let mut cost = vec![0.0; n * m];
        for i in 0..n {
            let xi = &x[i * dim..(i + 1) * dim];
            for j in 0..m {
                let yj = &y[j * dim..(j + 1) * dim];
                let mut dist_sq = 0.0;
                for k in 0..dim {
                    let diff = xi[k] - yj[k];
                    dist_sq += diff * diff;
                }
                cost[i * m + j] = dist_sq;
            }
        }
        cost
    }
}

/// Compute Euclidean cost matrix between two point clouds.
pub fn euclidean_cost_matrix(x: &[f64], y: &[f64], dim: usize) -> Vec<f64> {
    let mut cost = sqeuclidean_cost_matrix(x, y, dim);
    cost.par_iter_mut().for_each(|c| *c = c.sqrt());
    cost
}

/// Compute cost matrix between two 1D point sets with Minkowski metric.
pub fn minkowski_cost_matrix_1d(x: &[f64], y: &[f64], p: f64) -> Vec<f64> {
    let n = x.len();
    let m = y.len();

    if n >= PARALLEL_THRESHOLD {
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (0..m)
                    .map(|j| (x[i] - y[j]).abs().powf(p))
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        let mut cost = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                cost[i * m + j] = (x[i] - y[j]).abs().powf(p);
            }
        }
        cost
    }
}

/// Compute cosine distance cost matrix between two point clouds.
pub fn cosine_cost_matrix(x: &[f64], y: &[f64], dim: usize) -> Vec<f64> {
    let n = x.len() / dim;
    let m = y.len() / dim;

    // Precompute norms
    let x_norms: Vec<f64> = (0..n)
        .map(|i| {
            let xi = &x[i * dim..(i + 1) * dim];
            xi.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect();

    let y_norms: Vec<f64> = (0..m)
        .map(|j| {
            let yj = &y[j * dim..(j + 1) * dim];
            yj.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect();

    if n >= PARALLEL_THRESHOLD {
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let xi = &x[i * dim..(i + 1) * dim];
                (0..m)
                    .map(|j| {
                        let yj = &y[j * dim..(j + 1) * dim];
                        let dot: f64 = xi.iter().zip(yj.iter()).map(|(a, b)| a * b).sum();
                        let denom = x_norms[i] * y_norms[j];
                        if denom > 0.0 {
                            1.0 - dot / denom
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        let mut cost = vec![0.0; n * m];
        for i in 0..n {
            let xi = &x[i * dim..(i + 1) * dim];
            for j in 0..m {
                let yj = &y[j * dim..(j + 1) * dim];
                let dot: f64 = xi.iter().zip(yj.iter()).map(|(a, b)| a * b).sum();
                let denom = x_norms[i] * y_norms[j];
                cost[i * m + j] = if denom > 0.0 { 1.0 - dot / denom } else { 1.0 };
            }
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqeuclidean_2d() {
        let x = vec![0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let cost = sqeuclidean_cost_matrix(&x, &y, 2);
        assert_eq!(cost.len(), 4);
        assert!((cost[0] - 0.0).abs() < 1e-10);
        assert!((cost[1] - 2.0).abs() < 1e-10);
        assert!((cost[2] - 2.0).abs() < 1e-10);
        assert!((cost[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_2d() {
        let x = vec![0.0, 0.0];
        let y = vec![3.0, 4.0];
        let cost = euclidean_cost_matrix(&x, &y, 2);
        assert!((cost[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let cost = cosine_cost_matrix(&x, &y, 2);
        assert!((cost[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_same_direction() {
        let x = vec![1.0, 0.0];
        let y = vec![2.0, 0.0];
        let cost = cosine_cost_matrix(&x, &y, 2);
        assert!((cost[0] - 0.0).abs() < 1e-10);
    }
}
