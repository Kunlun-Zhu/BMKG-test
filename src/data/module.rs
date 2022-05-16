use crate::data::data::DataError;
use crate::data::data::TripleDataBatch;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Ix1;
use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::ToPyArray;
use pyo3::prelude::*;
use quick_error::quick_error;
use rand::prelude::*;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::sync::Arc;
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
    head_map: Arc<Vec<HashMap<i32, Vec<i32>>>>,
    #[pyo3(get, name="head_map")]
    /// Dict[(h, r), tensor.Tensor([t])]
    head_map_py: PyObject,
    #[pyo3(get, name="tail_map")]
    /// Dict[(t, r), tensor.Tensor([h])]
    tail_map_py: PyObject,

    // TripleDataModule
    // module: PyObject,
    // argparse.Namespace
    // config: PyObject,
    ent_size: usize,
    neg_sample_method: NegSampleMethod,
    train_neg_sample: usize,
    // rt: Runtime,
    // rank_handles: Vec<JoinHandle<Vec<i32>>>,
    // _marker: PhantomPinned,
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
        .ok_or_else(|| DataError::new_err("Wrong dataset"))?;
    Ok(())
}

#[allow(clippy::try_err)]
#[pymethods]
impl _TripleDataModule {
    #[new]
    fn new(module: &PyAny, py: Python, config: &PyAny) -> PyResult<Self> {
        let ent_size: usize = config.getattr("ent_size")?.extract()?;
        let mut head_map: Vec<HashMap<i32, Vec<i32>>> = Vec::default();
        head_map.resize(ent_size, HashMap::default());
        let mut tail_map: Vec<HashMap<i32, Vec<i32>>> = Vec::default();
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
        let torch = PyModule::import(py, "torch")?;
        let head_map_py: HashMap<_, _> = head_map
            .iter()
            .enumerate()
            .flat_map(|(h, m)| {
                m.iter().map(move |(r, t)| -> PyResult<((i64, i64), &PyAny)> {
                    Ok(
                        (
                            (h as i64, *r as i64),
                            torch.getattr("from_numpy")?
                                .call1((Array1::from_iter(t.iter().map(|i| *i as i64)).to_pyarray(py), ))?
                                .getattr("cuda")?
                                .call0()?
                            ,
                        )
                    )
                })
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        let tail_map_py: HashMap<_, _> = tail_map
            .iter()
            .enumerate()
            .flat_map(|(t, m)| {
                m.iter().map(move |(r, h)| -> PyResult<((i64, i64), &PyAny)> {
                    Ok(
                        (
                            (t as i64, *r as i64),
                            torch.getattr("from_numpy")?
                                .call1((Array1::from_iter(h.iter().map(|i| *i as i64)).to_pyarray(py), ))?
                                .getattr("cuda")?
                                .call0()?
                            ,
                        )
                    )
                })
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        // TODO: Add negative sample method configuration
        Ok(_TripleDataModule {
            head_map: Arc::new(head_map),
            head_map_py: head_map_py.into_py(py),
            tail_map_py: tail_map_py.into_py(py),
            // config: config.into_py(py),
            // module: module.into_py(py),
            neg_sample_method: NegSampleMethod::Random(RandomMethod::Tail),
            ent_size: config.getattr("ent_size")?.extract()?,
            train_neg_sample: config.getattr("train_neg_sample")?.extract()?,
            // rt: Runtime::new()?,
            // rank_handles: Vec::default(),
            // _marker: PhantomPinned,
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
                            for j in 0..self.train_neg_sample {
                                loop {
                                    idx = rng.gen_range(0..self.ent_size as i32);
                                    if !_TripleDataModule::has_edge(
                                        self.head_map.clone(),
                                        *h.get((i, 0))?,
                                        *r.get((i, 0))?,
                                        idx,
                                    ) {
                                        break;
                                    }
                                }
                                *row.get_mut(j)? = idx;
                            }
                            Some(())
                        })
                        .ok_or_else(|| {
                            DataError::new_err("error occured reading positive samples")
                        })?;
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
                            for j in 0..self.train_neg_sample {
                                loop {
                                    idx = rng.gen_range(0..self.ent_size as i32);
                                    if !_TripleDataModule::has_edge(
                                        self.head_map.clone(),
                                        idx,
                                        *r.get((i, 0))?,
                                        *t.get((i, 0))?,
                                    ) {
                                        break;
                                    }
                                }
                                *row.get_mut(j)? = idx;
                            }
                            Some(())
                        })
                        .ok_or_else(|| {
                            DataError::new_err("error occured reading positive samples")
                        })?;
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

    // fn calc_rank(
    //     &mut self,
    //     py: Python,
    //     mode: &str,
    //     batch: &TripleDataBatch,
    //     pos_score: &PyArray1<f32>, // (batch_size)
    //     neg_score: &PyArray2<f32>, // (batch_size, ent_size)
    // ) -> Result<(), DataModuleError> {
    //     match mode {
    //         "head" => {
    //             let r = batch
    //                 .r
    //                 .as_ref(py)
    //                 .to_owned_array()
    //                 .into_dimensionality::<Ix1>()?;
    //             let t = batch
    //                 .t
    //                 .as_ref(py)
    //                 .to_owned_array()
    //                 .into_dimensionality::<Ix1>()?;
    //             let pos_score = pos_score.to_owned_array();
    //             let neg_score = neg_score.to_owned_array();
    //             self.rank_handles
    //                 .push(self.rt.spawn(_TripleDataModule::_calc_h_rank(
    //                     self.head_map.clone(),
    //                     t,
    //                     r,
    //                     pos_score,
    //                     neg_score,
    //                 )));
    //             Ok(())
    //         }
    //         "tail" => {
    //             let h = batch
    //                 .h
    //                 .as_ref(py)
    //                 .to_owned_array()
    //                 .into_dimensionality::<Ix1>()?;
    //             let r = batch
    //                 .r
    //                 .as_ref(py)
    //                 .to_owned_array()
    //                 .into_dimensionality::<Ix1>()?;
    //             let pos_score = pos_score.to_owned_array();
    //             let neg_score = neg_score.to_owned_array();
    //             self.rank_handles
    //                 .push(self.rt.spawn(_TripleDataModule::_calc_t_rank(
    //                     self.head_map.clone(),
    //                     h,
    //                     r,
    //                     pos_score,
    //                     neg_score,
    //                 )));
    //             Ok(())
    //         }
    //         _ => Err(DataError::new_err(
    //             "Wrong mode: only head and tail are supported",
    //         ))?,
    //     }
    // }

    // fn gather_ranks<'py>(
    //     &mut self,
    //     py: Python<'py>,
    // ) -> Result<&'py PyArray1<i32>, DataModuleError> {
    //     // Take rank handles out
    //     let mut handles = Vec::new();
    //     swap(&mut self.rank_handles, &mut handles);
    //     let result = handles
    //         .into_iter()
    //         .map(|handle| self.rt.block_on(handle))
    //         .collect::<Result<Vec<_>, JoinError>>()?;
    //     let result: Vec<_> = result.into_iter().flatten().collect();

    //     let result_arr: Array1<i32> = Array1::from_vec(result);
    //     Ok(result_arr.into_pyarray(py))
    // }
}

impl _TripleDataModule {
    fn has_edge(head_map: Arc<Vec<HashMap<i32, Vec<i32>>>>, h: i32, r: i32, t: i32) -> bool {
        head_map[h as usize].contains_key(&r) && head_map[h as usize][&r].binary_search(&t).is_ok()
    }

    // async fn _calc_t_rank(
    //     head_map: Arc<Vec<HashMap<i32, Vec<i32>>>>,
    //     h: Array1<i32>,
    //     r: Array1<i32>,
    //     pos_score: Array1<f32>,
    //     neg_score: Array2<f32>,
    // ) -> Vec<i32> {
    //     let mut rank = Vec::with_capacity(pos_score.len());
    //     // calculate the rank of pos_score in neg_scores.
    //     for i in 0..pos_score.len() {
    //         let mut idx = 0;
    //         for j in 0..neg_score.shape()[1] {
    //             if pos_score[i] < neg_score[(i, j)]
    //                 && !_TripleDataModule::has_edge(head_map.clone(), h[i], r[i], j as i32)
    //             {
    //                 idx += 1;
    //             }
    //         }
    //         rank.push(idx);
    //     }
    //     rank
    // }
    // async fn _calc_h_rank(
    //     head_map: Arc<Vec<HashMap<i32, Vec<i32>>>>,
    //     t: Array1<i32>,
    //     r: Array1<i32>,
    //     pos_score: Array1<f32>,
    //     neg_score: Array2<f32>,
    // ) -> Vec<i32> {
    //     let mut rank = Vec::with_capacity(pos_score.len());
    //     // calculate the rank of pos_score in neg_scores.
    //     for i in 0..pos_score.shape()[0] {
    //         let mut idx = 0;
    //         for j in 0..neg_score.shape()[1] {
    //             if pos_score[i] < neg_score[(i, j)]
    //                 && !_TripleDataModule::has_edge(head_map.clone(), j as i32, r[i], t[i])
    //             {
    //                 idx += 1;
    //             }
    //         }
    //         rank.push(idx);
    //     }
    //     rank
    // }
}
