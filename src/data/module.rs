use crate::data::data::DataError;
use crate::data::data::TripleDataBatch;
use log::info;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::IntoDimension;
use ndarray::Ix1;
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use quick_error::quick_error;
use rand::prelude::*;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::{collections::HashMap, hash::Hash};

quick_error! {
    #[derive(Debug)]
    pub enum DataModuleError {
        Shape(err: ndarray::ShapeError) {
            source(err)
            from()
        }
        Python(err: pyo3::PyErr) {
            source(err)
            from()
        }
    }
}

impl std::convert::From<DataModuleError> for PyErr {
    fn from(err: DataModuleError) -> PyErr {
        match err {
            DataModuleError::Python(err) => err,
            _ => DataError::new_err(format!("{}:{} {}", file!(), line!(), err)),
        }
    }
}

enum RandomMethod {
    Head,
    Tail,
}

enum NegSampleMethod {
    Random(RandomMethod),
}

#[pyclass(module = "bmkg._data", subclass)]
pub struct _TripleDataModule {
    // (h, r) -> t
    head_map: Vec<HashMap<i32, Vec<i32>>>,
    // (t, r) -> h
    tail_map: Vec<HashMap<i32, Vec<i32>>>,
    // TripleDataModule
    module: PyObject,
    // argparse.Namespace
    config: PyObject,
    ent_size: usize,
    neg_sample_method: NegSampleMethod,
    train_neg_sample: usize,
}

fn handle_set(
    head_map: &mut Vec<HashMap<i32, Vec<i32>>>,
    tail_map: &mut Vec<HashMap<i32, Vec<i32>>>,
    set: &PyArray2<i32>,
) -> PyResult<()> {
    set.readonly()
        .as_array()
        .axis_iter(Axis(0))
        .try_for_each(|t| -> Option<()> {
            unsafe { head_map.get_unchecked_mut(*t.get(0)? as usize) }
                .entry(*t.get(1)?)
                .or_default()
                .push(*t.get(2)?);
            unsafe { tail_map.get_unchecked_mut(*t.get(2)? as usize) }
                .entry(*t.get(1)?)
                .or_default()
                .push(*t.get(0)?);
            Some(())
        })
        .ok_or(DataError::new_err("Wrong dataset"))?;
    Ok(())
}

#[pymethods]
impl _TripleDataModule {
    #[new]
    fn new(py: Python, module: &PyAny, config: &PyAny) -> PyResult<Self> {
        let mut head_map: Vec<HashMap<i32, Vec<i32>>> = Vec::default();
        let mut tail_map: Vec<HashMap<i32, Vec<i32>>> = Vec::default();
        let ent_size: usize = config.getattr("ent_size")?.extract()?;
        head_map.resize(ent_size, HashMap::default());
        tail_map.resize(ent_size, HashMap::default());
        let set: &PyArray2<i32> = module.getattr("train_set")?.getattr("data")?.extract()?;
        handle_set(&mut head_map, &mut tail_map, set)?;
        if module.getattr("valid_set").is_ok() {
            let set: &PyArray2<i32> = module.getattr("valid_set")?.getattr("data")?.extract()?;
            handle_set(&mut head_map, &mut tail_map, set)?;
        }
        if module.getattr("test_set").is_ok() {
            let set: &PyArray2<i32> = module.getattr("test_set")?.getattr("data")?.extract()?;
            handle_set(&mut head_map, &mut tail_map, set)?;
        }
        head_map.par_iter_mut().for_each(|map| {
            map.iter_mut().for_each(|t| {
                t.1.sort_unstable();
                t.1.dedup();
            })
        });

        // TODO: Add negative sample method configuration
        Ok(_TripleDataModule {
            head_map,
            tail_map,
            config: config.into_py(py),
            module: module.into_py(py),
            neg_sample_method: NegSampleMethod::Random(RandomMethod::Tail),
            ent_size: config.getattr("ent_size")?.extract()?,
            train_neg_sample: config.getattr("train_neg_sample")?.extract()?,
        })
    }

    fn neg_sample(
        &mut self,
        py: Python,
        batch: &TripleDataBatch,
    ) -> Result<TripleDataBatch, DataModuleError> {
        if batch.h.as_ref(py).shape().len() != 1 {
            return Err(DataError::new_err(
                "Wrong data: positive data should be 1-d array",
            ))?;
        }
        if batch.r.as_ref(py).shape().len() != 1 {
            return Err(DataError::new_err(
                "Wrong data: positive data should be 1-d array",
            ))?;
        }
        if batch.t.as_ref(py).shape().len() != 1 {
            return Err(DataError::new_err(
                "Wrong data: positive data should be 1-d array",
            ))?;
        }
        let h = batch.h.as_ref(py).to_owned_array();
        let r = batch.r.as_ref(py).to_owned_array();
        let t = batch.t.as_ref(py).to_owned_array();
        let len = h.len();
        let result;
        match &self.neg_sample_method {
            NegSampleMethod::Random(rand) => match rand {
                RandomMethod::Tail => {
                    // randomly replace tail entities.
                    let h = h.into_dimensionality::<Ix1>()?.into_shape((len, 1))?;
                    let r = r.into_dimensionality::<Ix1>()?.into_shape((len, 1))?;
                    let mut arr = Array2::zeros((len, self.train_neg_sample));
                    arr.axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .try_for_each(|(i, mut row)| -> Option<()> {
                            let mut rng = rand::thread_rng();
                            let mut idx;
                            for j in (0..self.train_neg_sample) {
                                loop {
                                    idx = rng.gen_range(0..self.ent_size as i32);
                                    match self.head_map[*h.get((i, 0))? as usize].get(r.get((i, 0))?) {
                                        Some(vv) => {
                                            if vv.binary_search(&idx).is_err() {
                                                break
                                            }
                                        },
                                        None => break,
                                    }
                                }
                                *row.get_mut(j)? = idx;
                            }
                            Some(())
                        })
                        .ok_or(DataError::new_err("error occured reading positive samples"))?;
                    self.neg_sample_method = NegSampleMethod::Random(RandomMethod::Head);
                    result = TripleDataBatch {
                        h: h.into_dyn().into_pyarray(py).into_py(py),
                        r: r.into_dyn().into_pyarray(py).into_py(py),
                        t: arr.into_dyn().into_pyarray(py).into_py(py),
                    };
                }
                RandomMethod::Head => {
                    // randomly replace head entities.
                    let t = t.into_dimensionality::<Ix1>()?.into_shape((len, 1))?;
                    let r = r.into_dimensionality::<Ix1>()?.into_shape((len, 1))?;
                    let mut arr = Array2::zeros((len, self.train_neg_sample));
                    arr.axis_iter_mut(Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .try_for_each(|(i, mut row)| -> Option<()> {
                            let mut rng = rand::thread_rng();
                            let mut idx;
                            for j in (0..self.train_neg_sample) {
                                loop {
                                    idx = rng.gen_range(0..self.ent_size as i32);
                                    match self.tail_map[*t.get((i, 0))? as usize].get(r.get((i, 0))?) {
                                        Some(vv) => {
                                            if vv.binary_search(&idx).is_err() {
                                                break
                                            }
                                        },
                                        None => break,
                                    }
                                }
                                *row.get_mut(j)? = idx;
                            }
                            Some(())
                        })
                        .ok_or(DataError::new_err("error occured reading positive samples"))?;
                    self.neg_sample_method = NegSampleMethod::Random(RandomMethod::Tail);
                    result = TripleDataBatch {
                        h: arr.into_dyn().into_pyarray(py).into_py(py),
                        r: r.into_dyn().into_pyarray(py).into_py(py),
                        t: t.into_dyn().into_pyarray(py).into_py(py),
                    };
                }
            },
        }
        Ok(result)
    }
}
