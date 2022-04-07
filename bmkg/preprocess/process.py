import os
import pathlib
from argparse import Namespace
from os import path

import numpy as np
from tqdm import tqdm


def _parse_srd_format(data_format):
    if data_format == "hrt":
        return [0, 1, 2]
    if data_format == "htr":
        return [0, 2, 1]
    if data_format == "rht":
        return [1, 0, 2]
    if data_format == "rth":
        return [2, 0, 1]
    if data_format == "thr":
        return [1, 2, 0]
    if data_format == "trh":
        return [2, 1, 0]


def get_vocab(vocab: dict[str, int], ent: str, atoi: list[str]):
    try:
        return vocab[ent]
    except KeyError:
        vocab[ent] = len(vocab)
        atoi.append(ent)
        assert len(atoi) == len(vocab)
        return vocab[ent]


def process(args: Namespace):
    # vocab
    data = {}
    vocab: dict[str, int] = {}
    atoi: list[str] = []
    ent_vocab: dict[str, int] = {}
    rel_vocab: dict[str, int] = {}
    ent_atoi: list[str] = []
    rel_atoi: list[str] = []
    data_format = _parse_srd_format(args.data_format)
    for file in tqdm(args.data_files):
        triples = []
        with open(path.join(args.data_path, file)) as f:
            for triple in tqdm(f):
                triple = triple.split(maxsplit=3)
                h, r, t = triple[data_format[0]], triple[data_format[1]], triple[data_format[2]]
                if args.union_vocab:
                    h, r, t = get_vocab(vocab, h, atoi), get_vocab(vocab, r, atoi), get_vocab(vocab, t, atoi)
                else:
                    h, t = get_vocab(ent_vocab, h, ent_atoi), get_vocab(ent_vocab, t, ent_atoi)
                    r = get_vocab(rel_vocab, r, rel_atoi)
                triples.append((h, r, t))
        data[file] = triples
    # TODO: run graph partition algorithm
    if args.partition != 1:
        raise NotImplementedError("Graph partitioning isn't currently supported.")
    output_path = args.output_path
    if output_path == "[DEFAULT]":
        output_path = path.join('data', path.split(args.data_path)[-1])
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    if args.union_vocab:
        with open(path.join(output_path, 'vocab.txt'), "w") as f:
            f.write("\n".join(map(lambda v: f"{v[1]}\t{v[0]}", enumerate(atoi))))
    else:
        with open(path.join(output_path, 'ent_vocab.txt'), "w") as f:
            f.write("\n".join(map(lambda v: f"{v[1]}\t{v[0]}", enumerate(ent_atoi))))
        with open(path.join(output_path, 'rel_vocab.txt'), "w") as f:
            f.write("\n".join(map(lambda v: f"{v[1]}\t{v[0]}", enumerate(rel_atoi))))
    for file in tqdm(args.data_files):
        data[file] = np.asarray(data[file])
        np.save(pathlib.Path(output_path) / path.splitext(file)[0], data[file])
