use pyo3::prelude::*;
mod data;
mod preprocess;

/// A Python module implemented in Rust.
#[pymodule]
fn bmkg(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    preprocess::register(_py, m)?;
    data::register(_py, m)?;
    Ok(())
}
