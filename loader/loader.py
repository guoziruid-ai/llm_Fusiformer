import os
import json
import zipfile
import requests
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
import dgl
from transformers import BertModel, BertTokenizer

from .utils import get_db_info
BERT_PATH = 'my_bert_base_uncased'

def split_train_val_test(
    total_size,
    ratio_train_val_test=None,
    n_train_val_test=None,
    keep_order=False,
    split_seed=7
):
    if ratio_train_val_test:
        assert len(ratio_train_val_test) == 3
        ratio_train, ratio_val, ratio_test = ratio_train_val_test
        n_train = int(ratio_train * total_size)
        n_val = int(ratio_val * total_size) if ratio_val else None
        n_test = int(ratio_test * total_size) if ratio_test else None
    elif n_train_val_test:
        assert len(n_train_val_test) == 3
        n_train, n_val, n_test = n_train_val_test
    else:
        raise ValueError("Please Specify the dataset division.")

    ids = list(np.arange(total_size))
    if not keep_order:
        random.seed(split_seed)
        random.shuffle(ids)

    train_idx = ids[:n_train]
    val_idx = ids[n_train: n_train+n_val] if n_val else None
    test_idx = ids[-n_test:] if n_test else None
    return train_idx, val_idx, test_idx


def get_dataset(
    data_path, 
    ratio_train_val_test=None, 
    n_train_val_test=None,
    transforms=None,
    graph=False, 
    line_graph=False
):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    print("Loading completed.")

    train_idx, val_idx, test_idx = split_train_val_test(
                                        total_size=len(data),
                                        ratio_train_val_test=ratio_train_val_test,
                                        n_train_val_test=n_train_val_test)
    train_data = [data[idx] for idx in train_idx]
    val_data = [data[idx] for idx in val_idx] if val_idx else None
    test_data = [data[idx] for idx in test_idx] if val_idx else None

    if isinstance(transforms, list): 
        assert len(transforms) <= 3 
        if len(transforms) == 3: 
            train_transform, val_transform, test_transform = transforms
        elif len(transforms) == 2: 
            train_transform = transforms[0]
            val_transform = test_transform = transforms[-1]
        else:
            train_transform = val_transform = test_transform = transforms
    else:
        train_transform = val_transform = test_transform = transforms

    if graph:
        train_dataset = CrystalGraphDataset(data=train_data, transforms=train_transform, line_graph=line_graph, text_tokenizer=BertTokenizer.from_pretrained(BERT_PATH))
        val_dataset = CrystalGraphDataset(data=val_data, transforms=val_transform, line_graph=line_graph, text_tokenizer=BertTokenizer.from_pretrained(BERT_PATH)) if val_data else None
        test_dataset = CrystalGraphDataset(data=test_data, transforms=test_transform, line_graph=line_graph, text_tokenizer=BertTokenizer.from_pretrained(BERT_PATH)) if test_data else None
    else:
        train_dataset = CrystalDataset(data=train_data, transforms=train_transform)
        val_dataset = CrystalDataset(data=val_data, transforms=val_transform) if val_data else None
        test_dataset = CrystalDataset(data=test_data, transforms=test_transform) if test_data else None

    return train_dataset, val_dataset, test_dataset


class CrystalGraphDataset(Dataset):
    def __init__(self, data, transforms, cutoff=8, max_neighbors=12, line_graph=False,
                    text_tokenizer=None, text_max_length=512, text_field='description'):
        self.data = data
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.transforms = transforms
        self.line_graph = line_graph

        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.text_field = text_field

    def __getitem__(self, index):
        info = self.data[index]

        crystal = dict()
        crystal['info'] = info
        crystal['structure'] = Atoms.from_dict(info['atoms'])
        crystal['graph'] = self.build_graph(crystal['structure'])
        if self.line_graph:
            crystal['line_graph'] = self.build_line_graph(crystal['graph'])

        if self.text_tokenizer is not None:
            text = info.get(self.text_field, "")
            tokens = self.text_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.text_max_length,
                return_tensors='pt'
            )
            crystal['text'] = tokens
        inputs, targets = self.transforms(crystal)
        print(type(inputs))
        return inputs, targets

    def __len__(self):
        return len(self.data)
    
    def build_graph(self, atoms):
        edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=True,
            )
        u, v, r = build_undirected_edgedata(atoms, edges)
        g = dgl.graph((u, v))
        g.edata['offset'] = r
        return g

    def build_line_graph(self, g):
        lg = g.line_graph(shared=True)
        return lg

    def collect(self):
        def collect_line_graph(samples):
            inputs, targets = map(list, zip(*samples))
            graphs = [item['graph_input'] for item in inputs]

            batched_graph = dgl.batch([g[0] for g in graphs])
            batched_line_graph = dgl.batch([g[1] for g in graphs])

            text_embs = [item.get('text_emb', None) for item in inputs]
            if all(emb is not None for emb in text_embs):
                batched_text_emb = torch.stack(text_embs)
            else:
                batched_text_emb = None

            batched_inputs = {
            'graph_input': [batched_graph, batched_line_graph]
            }
            if batched_text_emb is not None:
                batched_inputs['text_emb'] = batched_text_emb
        
            return batched_inputs, torch.stack(targets)

        return collect_line_graph