use numpy::PyArray1;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

/// TripleDataBatch is a batch of triple data <h,r,t>, stored in 3 numpy arrays.
#[pyclass(module = "bmkg._data", subclass)]
#[pyo3(text_signature = "(h, r, t, /)")]
pub struct TripleDataBatch {
    #[pyo3(get, set)]
    h: Py<PyArray1<i32>>,
    #[pyo3(get, set)]
    r: Py<PyArray1<i32>>,
    #[pyo3(get, set)]
    t: Py<PyArray1<i32>>,
}

create_exception!(bmkg, DataError, PyException);

#[pymethods]
impl TripleDataBatch {
    #[new]
    fn new(py: Python, h: &PyArray1<i32>, r: &PyArray1<i32>, t: &PyArray1<i32>) -> PyResult<Self> {
        if h.shape() != r.shape() || r.shape() != t.shape() {
            return Err(DataError::new_err(format!(
                "h, r, t has differnet shape\n h: {:?}, r: {:?}, t: {:?}",
                h.shape(),
                r.shape(),
                t.shape()
            )));
        }
        Ok(TripleDataBatch {
            h: (h.into_py(py)),
            r: (r.into_py(py)),
            t: (t.into_py(py)),
        })
    }
}
