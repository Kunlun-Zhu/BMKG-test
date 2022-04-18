use pyo3::prelude::*;
#[allow(clippy::module_inception)]
mod data;
mod module;

pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "_data")?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bmkg._data", m)?;
    m.add_class::<data::TripleDataBatch>()?;
    m.add_class::<module::_TripleDataModule>()?;
    parent_module.add_submodule(m)?;
    Ok(())
}
