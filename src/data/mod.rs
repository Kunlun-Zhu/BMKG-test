use pyo3::prelude::*;

mod data;

pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "_data")?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bmkg._data", m)?;
    m.add_class::<data::TripleDataBatch>()?;
    parent_module.add_submodule(m)?;
    Ok(())
}
