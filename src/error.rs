use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

/// Errors that can occur in kantorovich computations.
#[derive(Error, Debug)]
pub enum KantorovichError {
    #[error("Sinkhorn did not converge after {iterations} iterations (tolerance: {tolerance:.2e}, achieved: {achieved:.2e})")]
    ConvergenceError {
        iterations: usize,
        tolerance: f64,
        achieved: f64,
    },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Numerical error: {message}")]
    NumericalError { message: String },
}

impl From<KantorovichError> for PyErr {
    fn from(err: KantorovichError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Validate that a probability vector is valid (non-negative, sums to ~1).
pub fn validate_probability_vector(v: &[f64], name: &str) -> Result<(), KantorovichError> {
    if v.is_empty() {
        return Err(KantorovichError::InvalidInput {
            message: format!("{name} must not be empty"),
        });
    }

    for (i, &val) in v.iter().enumerate() {
        if val.is_nan() {
            return Err(KantorovichError::InvalidInput {
                message: format!("{name}[{i}] is NaN"),
            });
        }
        if val < 0.0 {
            return Err(KantorovichError::InvalidInput {
                message: format!("{name}[{i}] = {val} is negative"),
            });
        }
    }

    let sum: f64 = v.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(KantorovichError::InvalidInput {
            message: format!("{name} sums to {sum}, expected ~1.0"),
        });
    }

    Ok(())
}

/// Validate that a cost matrix has the correct dimensions.
pub fn validate_cost_matrix(
    m: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<(), KantorovichError> {
    let expected = n_rows * n_cols;
    if m.len() != expected {
        return Err(KantorovichError::InvalidInput {
            message: format!(
                "Cost matrix has {} elements, expected {} ({n_rows} x {n_cols})",
                m.len(),
                expected
            ),
        });
    }

    for (i, &val) in m.iter().enumerate() {
        if val.is_nan() {
            return Err(KantorovichError::InvalidInput {
                message: format!("Cost matrix contains NaN at index {i}"),
            });
        }
        if val.is_infinite() {
            return Err(KantorovichError::InvalidInput {
                message: format!("Cost matrix contains Inf at index {i}"),
            });
        }
    }

    Ok(())
}
