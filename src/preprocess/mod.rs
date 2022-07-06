use std::collections::HashMap;
use std::fmt;
use std::fs::{create_dir_all, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::RwLock;

use lasso::{Key, Rodeo};
use log::info;
use metis::Graph;
use ndarray::prelude::*;
use numpy::ndarray::Axis;
use numpy::ToPyArray;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use quick_error::quick_error;
use rayon::prelude::*;
use serde_json::json;

quick_error! {
    #[derive(Debug)]
    pub enum PreprocessErrorWrap {
        Metis(err: metis::Error) {
            source(err)
            from()
        }
        Shape(err: ndarray::ShapeError) {
            source(err)
            from()
        }
        IO(err: std::io::Error){
            source(err)
            from()
        }
        Python(err: pyo3::PyErr) {
            source(err)
            from()
        }
    }
}

impl std::convert::From<PreprocessErrorWrap> for PyErr {
    fn from(err: PreprocessErrorWrap) -> PyErr {
        match err {
            PreprocessErrorWrap::Python(err) => err,
            _ => PreprocessError::new_err(err.to_string()),
        }
    }
}

fn argsort<T: Ord>(data: &[T]) -> Vec<i32> {
    let mut indices = (0..data.len() as i32).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i as usize]);
    indices
}

create_exception!(bmkg, PreprocessError, PyException);

fn parse_format(data_format: &str) -> Result<[usize; 3], PyErr> {
    match data_format {
        "hrt" => Ok([0, 1, 2]),
        "htr" => Ok([0, 2, 1]),
        "rht" => Ok([1, 0, 2]),
        "rth" => Ok([2, 0, 1]),
        "thr" => Ok([1, 2, 0]),
        "trh" => Ok([2, 1, 0]),
        _ => Err(PreprocessError::new_err("Invalid data format")),
    }
}
#[derive(PartialEq, Eq, Copy, Clone)]
struct StrKey(i32);

unsafe impl Key for StrKey {
    fn into_usize(self) -> usize {
        self.0 as usize
    }

    fn try_from_usize(int: usize) -> Option<Self> {
        if int > i32::MAX as usize {
            None
        } else {
            Some(StrKey(int as i32))
        }
    }
}

#[pyfunction]
#[pyo3(text_signature = "(args, /)")]
/// Run preprocess by given arguments.
///
/// :param args: argparse.Namespace
/// :return: None
fn preprocess(py: Python, args: &PyAny) -> Result<(), PreprocessErrorWrap> {
    let mut data = HashMap::<&str, Array2<i32>>::new();
    let mut ent_vocab = Rodeo::<StrKey>::new();
    let mut rel_vocab = Rodeo::<StrKey>::new();
    let data_format = parse_format(args.getattr("data_format")?.extract()?)?;
    let files: Vec<&str> = args.getattr("data_files")?.extract()?;
    let path = PathBuf::from(args.getattr("data_path")?.extract::<&str>()?);
    for file_name in &files {
        info!("Processing {file_name}");
        let mut triples = Array2::<i32>::zeros((0, 3));
        let file = BufReader::new(File::open(path.join(file_name))?);
        for line in file.lines() {
            let line = line?;
            let triple: Vec<&str> = line.split(&[' ', '\t', ','][..]).collect();
            if triple.len() != 3 {
                return Err(PreprocessErrorWrap::Python(PreprocessError::new_err(
                    "Wrong data format! Expected {data_format}, got {line}",
                )));
            }
            let h = triple[data_format[0]];
            let r = triple[data_format[1]];
            let t = triple[data_format[2]];
            let h = ent_vocab.get_or_intern(h);
            let r = rel_vocab.get_or_intern(r);
            let t = ent_vocab.get_or_intern(t);
            triples.push_row(ArrayView::from(&[h.0, r.0, t.0]))?;
        }
        info!("{file_name} has {} triples!", triples.shape()[0]);
        data.insert(file_name, triples);
    }
    info!(
        "Read {} entities, {} relations.",
        ent_vocab.len(),
        rel_vocab.len()
    );
    let n_part = args.getattr("partition")?.extract::<u32>()?;
    let ids_map;
    let id_map: Box<dyn Fn(i32) -> i32> = if n_part != 1 {
        ids_map = graph_partition(data[files[0]].view(), &ent_vocab, n_part as i32)?;
        for graph in data.values_mut() {
            graph
                .slice_mut(s![.., 0 as usize])
                .par_mapv_inplace(|x| return ids_map[x as usize]);
            graph
                .slice_mut(s![.., 2 as usize])
                .par_mapv_inplace(|x| return ids_map[x as usize]);
        }
        Box::new(|x| ids_map[x as usize])
    } else {
        Box::new(|x| x)
    };
    let mut output_path = PathBuf::from(args.getattr("output_path")?.extract::<&str>()?);
    if output_path.as_os_str() == "[DEFAULT]" {
        output_path = PathBuf::from("./data").join(path.file_name().unwrap());
    }
    info!("Writing to {:?}", output_path.as_os_str().to_str());
    create_dir_all(&output_path)?;
    if args.getattr("union_vocab")?.extract::<bool>()? {
        let mut vocab_file = File::create(output_path.join("vocab.txt"))?;
        ent_vocab
            .iter()
            .try_for_each(|(k, v)| -> Result<(), io::Error> {
                let x = format!("{} {}\n", v, k.0);
                vocab_file.write(x.as_bytes()).and(Ok(()))
            })?;
        rel_vocab
            .iter()
            .try_for_each(|(k, v)| -> Result<(), io::Error> {
                let x = format!("{} {}\n", v, k.0);
                vocab_file.write(x.as_bytes()).and(Ok(()))
            })?;
        // union vocab requires adding ent_vocab.len() to all relation embeddings.
        for graph in data.values_mut() {
            graph
                .slice_mut(s![.., 1 as usize])
                .par_mapv_inplace(|x| x + ent_vocab.len() as i32);
        }
    } else {
        let mut ent_vocab_file = File::create(output_path.join("ent_vocab.txt"))?;
        let mut rel_vocab_file = File::create(output_path.join("rel_vocab.txt"))?;
        ent_vocab
            .iter()
            .try_for_each(|(k, v)| -> Result<(), io::Error> {
                let x = format!("{} {}\n", v, id_map(k.0));
                ent_vocab_file.write(x.as_bytes()).and(Ok(()))
            })?;
        rel_vocab
            .iter()
            .try_for_each(|(k, v)| -> Result<(), io::Error> {
                let x = format!("{} {}\n", v, id_map(k.0) + ent_vocab.len() as i32);
                rel_vocab_file.write(x.as_bytes()).and(Ok(()))
            })?;
    }
    let numpy = PyModule::import(py, "numpy")?;
    for (k, v) in data {
        // call numpy.save
        let py_array = v.to_pyarray(py);
        let mut path = output_path.join(k);
        path.set_extension("npy");
        let path = path.to_str().unwrap();
        numpy.getattr("save")?.call1((path, py_array))?;
    }
    let config = json!({
        "ent_size": ent_vocab.len(),
        "rel_size": rel_vocab.len(),
    });
    let mut config_file = File::create(output_path.join("config.json"))?;
    write!(config_file, "{}", config)?;
    Ok(())
}

fn graph_partition(
    graph: ArrayView2<i32>,
    ent_atoi: &Rodeo<StrKey>,
    n_part: i32,
) -> Result<Vec<i32>, PreprocessErrorWrap> {
    if graph.shape()[1] != 3 {
        return Err(PreprocessErrorWrap::Python(PyTypeError::new_err(
            "data['file_name'] should be a np.array with shape of (n_ent, 3)",
        )));
    }
    let part_size = (ent_atoi.len() as f64 / n_part as f64).floor() as i32;
    info!("Vocab size: {}", ent_atoi.len());
    info!("part_size: {}", part_size);
    info!("Triple count: {}", graph.shape()[0]);
    let count: usize = graph
        .axis_iter(Axis(0))
        .filter_map(|x| -> Option<_> { Some(x.get(0)? / part_size != x.get(2)? / part_size) })
        .filter(|x| *x)
        .count();
    info!(
        "Before partition: there are about {}({:.1}%) cross-part edges.",
        count,
        count as f64 / graph.shape()[0] as f64 * 100.0
    );
    // adjacency nodes
    let adjs: Vec<_> = (0..ent_atoi.len())
        .map(|_| -> _ { RwLock::<Vec<i32>>::default() })
        .collect();
    graph
        .axis_iter(Axis(0))
        .into_par_iter()
        .try_for_each(|x| -> Option<()> {
            // get adjs[x[0]]
            // The python part guranteed that x[0] is smaller than ent_atoi.len()
            unsafe { adjs.get_unchecked(*(x.get(0)?) as usize) }
                // unwrap it from RwLock
                .write()
                .ok()?
                // push x[2]
                .push(*(x.get(2)?));
            // vice-versa
            unsafe { adjs.get_unchecked(*(x.get(2)?) as usize) }
                .write()
                .ok()?
                .push(*(x.get(0)?));
            Some(())
        })
        .ok_or(PreprocessError::new_err(
            "Could not build graph! Shouldn't happen. Please report a bug at BMKG repo",
        ))?;
    info!("Adjacency nodes calculated! Generating xadj data...");
    let adjs: Vec<_> = adjs
        .into_iter()
        .map(|x| {
            let mut t = x.into_inner().unwrap();
            t.sort();
            t.dedup();
            t
        })
        .collect();
    let mut xadj: Vec<_> = [0]
        .into_iter()
        .chain(adjs.iter().map(|x| x.len()).scan(0i32, |sum, i| {
            *sum += i as i32;
            Some(*sum)
        }))
        .collect();
    let mut adjncy = adjs.join(&[0][1..]);
    info!("Doing partition...");
    // All set!
    let mut part = vec![0; ent_atoi.len()];
    let cut = Graph::new(1i32, n_part, xadj.as_mut_slice(), adjncy.as_mut_slice())
        .set_option(metis::option::NSeps(2))
        .part_kway(part.as_mut_slice())?;
    // Run two argsort to get id_map
    let ids_map = argsort(&argsort(&part));
    for i in 0..n_part {
        let count = part.iter().filter(|x| **x == i).count();
        info!("Part {} has {} nodes", i, count);
    }
    // debug!("cut: {}", cut);
    // debug!("part: {:?}", &part);
    // debug!("ids_map: {:?}", &ids_map);
    // map the graph using ids_map
    // unsafe { graph.as_array_mut() }
    //     .slice_mut(s![.., 0])
    //     .par_mapv_inplace(|x| return ids_map[x as usize]);
    // unsafe { graph.as_array_mut() }
    //     .slice_mut(s![.., 2])
    //     .par_mapv_inplace(|x| return ids_map[x as usize]);
    let count: usize = graph
        .axis_iter(Axis(0))
        .filter_map(|x| -> Option<_> {
            Some(
                ids_map[*x.get(0)? as usize] / part_size
                    != ids_map[*x.get(2)? as usize] / part_size,
            )
        })
        .filter(|x| *x)
        .count();
    info!(
        "After partition: there are about {}({:.1}%) cross-part edges.",
        count,
        count as f64 / graph.shape()[0] as f64 * 100.0
    );
    Ok(ids_map)
}

pub fn register(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "_preprocess")?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bmkg._preprocess", m)?;
    m.add_wrapped(wrap_pyfunction!(preprocess))?;

    parent_module.add_submodule(m)?;
    Ok(())
}
