use numpy::PyArrayDyn;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

/// TripleDataBatch is a batch of triple data <h,r,t>, stored in 3 numpy arrays.
#[pyclass(module = "bmkg._data", subclass)]
#[pyo3(text_signature = "(h, r, t, /)")]
pub struct TripleDataBatch {
    #[pyo3(get, set)]
    pub(crate) h: Py<PyArrayDyn<i32>>,
    #[pyo3(get, set)]
    pub(crate) r: Py<PyArrayDyn<i32>>,
    #[pyo3(get, set)]
    pub(crate) t: Py<PyArrayDyn<i32>>,
}

create_exception!(bmkg, DataError, PyException);

#[pymethods]
impl TripleDataBatch {
    #[new]
    fn new(
        py: Python,
        h: &PyArrayDyn<i32>,
        r: &PyArrayDyn<i32>,
        t: &PyArrayDyn<i32>,
    ) -> PyResult<Self> {
        let numpy = PyModule::import(py, "numpy")?;
        if numpy.getattr("broadcast")?.call1((h, r, t)).is_err() {
            return Err(DataError::new_err(format!(
                "h, r, t should be broadcastable\n h: {:?}, r: {:?}, t: {:?}",
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
