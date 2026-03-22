pub mod barycenter;
pub mod cost;
pub mod divergence;
pub mod error;
pub mod exact_1d;
pub mod sinkhorn;
pub mod sliced;
pub mod unbalanced;

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Compute the Sinkhorn optimal transport plan and cost.
///
/// Args:
///     a: Source distribution (1D array, sums to 1)
///     b: Target distribution (1D array, sums to 1)
///     M: Cost matrix (2D array, shape (len(a), len(b)))
///     reg: Regularization parameter (default: 0.1)
///     max_iter: Maximum iterations (default: 1000)
///     tol: Convergence tolerance (default: 1e-8)
///     method: "standard" or "log" (default: "standard")
///
/// Returns:
///     dict with keys: "plan", "cost", "u", "v", "iterations"
#[pyfunction]
#[pyo3(signature = (a, b, m, reg=0.1, max_iter=1000, tol=1e-8, method="standard"))]
fn sinkhorn_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    m: PyReadonlyArrayDyn<'py, f64>,
    reg: f64,
    max_iter: usize,
    tol: f64,
    method: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    let m_slice = m.as_slice()?;

    let log_domain = match method {
        "standard" => false,
        "log" => true,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown method: '{method}'. Use 'standard' or 'log'."
            )))
        }
    };

    let result = py.detach(|| {
        sinkhorn::sinkhorn(a_slice, b_slice, m_slice, reg, max_iter, tol, log_domain)
    })?;

    let n = result.n_rows;
    let m_cols = result.n_cols;

    let dict = pyo3::types::PyDict::new(py);
    let plan_array = ArrayD::from_shape_vec(vec![n, m_cols], result.plan)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    dict.set_item("plan", plan_array.into_pyarray(py))?;
    dict.set_item("cost", result.cost)?;
    let u_array = ArrayD::from_shape_vec(vec![n], result.u)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    dict.set_item("u", u_array.into_pyarray(py))?;
    let v_array = ArrayD::from_shape_vec(vec![m_cols], result.v)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    dict.set_item("v", v_array.into_pyarray(py))?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict)
}

/// Compute the Wasserstein-p distance between two 1D distributions.
#[pyfunction]
#[pyo3(signature = (x_a, a, x_b, b, p=1.0))]
fn emd_1d(
    py: Python<'_>,
    x_a: PyReadonlyArrayDyn<'_, f64>,
    a: PyReadonlyArrayDyn<'_, f64>,
    x_b: PyReadonlyArrayDyn<'_, f64>,
    b: PyReadonlyArrayDyn<'_, f64>,
    p: f64,
) -> PyResult<f64> {
    let x_a_slice = x_a.as_slice()?;
    let a_slice = a.as_slice()?;
    let x_b_slice = x_b.as_slice()?;
    let b_slice = b.as_slice()?;

    let result = py.detach(|| {
        exact_1d::wasserstein_1d(x_a_slice, a_slice, x_b_slice, b_slice, p)
    })?;

    Ok(result)
}

/// Compute a cost matrix between two point clouds.
#[pyfunction]
#[pyo3(signature = (x, y, metric="sqeuclidean"))]
fn cost_matrix<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    metric: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let x_shape = x.shape().to_vec();
    let y_shape = y.shape().to_vec();

    if x_shape.len() != 2 || y_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must be 2D arrays (n_points, n_dims)",
        ));
    }
    if x_shape[1] != y_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x has {} dimensions but y has {}",
            x_shape[1], y_shape[1]
        )));
    }

    let n = x_shape[0];
    let m = y_shape[0];
    let dim = x_shape[1];
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;

    let cost_vec = py.detach(|| match metric {
        "sqeuclidean" => Ok(cost::sqeuclidean_cost_matrix(x_slice, y_slice, dim)),
        "euclidean" => Ok(cost::euclidean_cost_matrix(x_slice, y_slice, dim)),
        "cosine" => Ok(cost::cosine_cost_matrix(x_slice, y_slice, dim)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown metric: '{metric}'. Use 'sqeuclidean', 'euclidean', or 'cosine'."
        ))),
    })?;

    let array = ArrayD::from_shape_vec(vec![n, m], cost_vec)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Compute the Sinkhorn divergence (debiased OT cost).
///
/// SD(a, b) = S(a,b) - 0.5*S(a,a) - 0.5*S(b,b)
///
/// Args:
///     a, b: Distributions (1D arrays, sum to 1)
///     cost_ab: Cost matrix a vs b (2D)
///     cost_aa: Cost matrix a vs a (2D)
///     cost_bb: Cost matrix b vs b (2D)
///     reg: Regularization (default: 0.1)
///     max_iter: Max iterations (default: 1000)
///     tol: Tolerance (default: 1e-8)
///
/// Returns:
///     dict with "divergence", "cost_ab", "cost_aa", "cost_bb"
#[pyfunction]
#[pyo3(signature = (a, b, cost_ab, cost_aa, cost_bb, reg=0.1, max_iter=1000, tol=1e-8))]
fn sinkhorn_divergence_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    cost_ab: PyReadonlyArrayDyn<'py, f64>,
    cost_aa: PyReadonlyArrayDyn<'py, f64>,
    cost_bb: PyReadonlyArrayDyn<'py, f64>,
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let a_s = a.as_slice()?;
    let b_s = b.as_slice()?;
    let cab = cost_ab.as_slice()?;
    let caa = cost_aa.as_slice()?;
    let cbb = cost_bb.as_slice()?;

    let result = py.detach(|| {
        divergence::sinkhorn_divergence(a_s, b_s, cab, caa, cbb, reg, max_iter, tol)
    })?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("divergence", result.divergence)?;
    dict.set_item("cost_ab", result.cost_ab)?;
    dict.set_item("cost_aa", result.cost_aa)?;
    dict.set_item("cost_bb", result.cost_bb)?;
    Ok(dict)
}

/// Compute sliced Wasserstein distance via random 1D projections.
///
/// Scales to N=100,000+ (no NxN cost matrix needed).
///
/// Args:
///     x, y: Point clouds (2D arrays)
///     a, b: Distribution weights (1D arrays)
///     n_projections: Number of random projections (default: 50)
///     p: Wasserstein order (default: 2.0)
///     seed: Random seed (default: 42)
///
/// Returns:
///     Sliced Wasserstein distance (float)
#[pyfunction]
#[pyo3(signature = (x, y, a, b, n_projections=50, p=2.0, seed=42))]
fn sliced_wasserstein_solve(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, f64>,
    y: PyReadonlyArrayDyn<'_, f64>,
    a: PyReadonlyArrayDyn<'_, f64>,
    b: PyReadonlyArrayDyn<'_, f64>,
    n_projections: usize,
    p: f64,
    seed: u64,
) -> PyResult<f64> {
    let x_shape = x.shape().to_vec();
    let y_shape = y.shape().to_vec();

    if x_shape.len() != 2 || y_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must be 2D arrays (n_points, n_dims)",
        ));
    }
    if x_shape[1] != y_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x has {} dims but y has {}",
            x_shape[1], y_shape[1]
        )));
    }

    let dim = x_shape[1];
    let x_s = x.as_slice()?;
    let y_s = y.as_slice()?;
    let a_s = a.as_slice()?;
    let b_s = b.as_slice()?;

    let result = py.detach(|| {
        sliced::sliced_wasserstein(x_s, y_s, a_s, b_s, dim, n_projections, p, seed)
    })?;

    Ok(result)
}

/// Solve unbalanced OT with KL divergence penalty.
///
/// Handles distributions with different total mass.
///
/// Args:
///     a, b: Distributions (non-negative, need not sum to 1)
///     M: Cost matrix (2D)
///     reg: Entropic regularization (default: 0.1)
///     tau: KL penalty weight (default: 1.0, larger = closer to balanced)
///     max_iter: Max iterations (default: 1000)
///     tol: Tolerance (default: 1e-8)
///
/// Returns:
///     dict with "plan", "cost", "iterations"
#[pyfunction]
#[pyo3(signature = (a, b, m, reg=0.1, tau=1.0, max_iter=1000, tol=1e-8))]
fn sinkhorn_unbalanced_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f64>,
    b: PyReadonlyArrayDyn<'py, f64>,
    m: PyReadonlyArrayDyn<'py, f64>,
    reg: f64,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let a_s = a.as_slice()?;
    let b_s = b.as_slice()?;
    let m_s = m.as_slice()?;

    let result = py.detach(|| {
        unbalanced::sinkhorn_unbalanced(a_s, b_s, m_s, reg, tau, max_iter, tol)
    })?;

    let n = result.n_rows;
    let mc = result.n_cols;

    let dict = pyo3::types::PyDict::new(py);
    let plan_array = ArrayD::from_shape_vec(vec![n, mc], result.plan)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    dict.set_item("plan", plan_array.into_pyarray(py))?;
    dict.set_item("cost", result.cost)?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict)
}

/// Compute fixed-support Wasserstein barycenter.
///
/// Finds the distribution on a fixed support that minimizes the weighted
/// sum of Sinkhorn distances to the input distributions.
///
/// Args:
///     distributions: List of distributions (each a 1D array on same support)
///     cost: Cost matrix on the support (2D, n_support x n_support)
///     weights: Weights for each distribution (1D, sums to 1)
///     reg: Regularization (default: 0.1)
///     max_iter: Max iterations (default: 100)
///     tol: Tolerance (default: 1e-6)
///
/// Returns:
///     dict with "barycenter" (1D array), "iterations"
#[pyfunction]
#[pyo3(signature = (distributions, cost, weights, reg=0.1, max_iter=100, tol=1e-6))]
fn barycenter_solve<'py>(
    py: Python<'py>,
    distributions: Vec<PyReadonlyArrayDyn<'py, f64>>,
    cost: PyReadonlyArrayDyn<'py, f64>,
    weights: PyReadonlyArrayDyn<'py, f64>,
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let cost_s = cost.as_slice()?;
    let weights_s = weights.as_slice()?;

    let dist_vecs: Vec<Vec<f64>> = distributions
        .iter()
        .map(|d| d.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;

    let dist_refs: Vec<&[f64]> = dist_vecs.iter().map(|v| v.as_slice()).collect();

    let result = py.detach(|| {
        barycenter::free_support_barycenter(&dist_refs, cost_s, weights_s, reg, max_iter, tol)
    })?;

    let dict = pyo3::types::PyDict::new(py);
    let bary_array = ArrayD::from_shape_vec(vec![result.barycenter.len()], result.barycenter)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    dict.set_item("barycenter", bary_array.into_pyarray(py))?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict)
}

/// kantorovich: High-performance optimal transport in Rust.
#[pymodule]
mod kantorovich {
    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(sinkhorn_solve, m)?)?;
        m.add_function(wrap_pyfunction!(emd_1d, m)?)?;
        m.add_function(wrap_pyfunction!(cost_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(sinkhorn_divergence_solve, m)?)?;
        m.add_function(wrap_pyfunction!(sliced_wasserstein_solve, m)?)?;
        m.add_function(wrap_pyfunction!(sinkhorn_unbalanced_solve, m)?)?;
        m.add_function(wrap_pyfunction!(barycenter_solve, m)?)?;
        Ok(())
    }
}
