use numpy::PyArray2;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

#[pyclass]
struct TripleDataset {
    /// A numpy.array with shape (_, 3)
    data: &PyArray2<i32>,
    start: usize,
    end: usize
}
