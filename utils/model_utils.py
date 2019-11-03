import os
import time
from typing import Tuple, Type, Dict, Any

import pickle

from models import (Sparse_Graph_Model, GGNN_Model, GNN_FiLM_Model, GNN_Edge_MLP_Model,
                    RGAT_Model, RGCN_Model, RGDCN_Model, RGIN_Model)
from tasks import Sparse_Graph_Task, QM9_Task, Citation_Network_Task, PPI_Task, VarMisuse_Task


def name_to_task_class(name: str) -> Tuple[Type[Sparse_Graph_Task], Dict[str, Any]]:
    name = name.lower()
    if name == "qm9":
        return QM9_Task, {}
    if name == "cora":
        return Citation_Network_Task, {"data_kind": "cora"}
    if name == "citeseer":
        return Citation_Network_Task, {"data_kind": "citeseer"}
    if name == "pubmed":
        return Citation_Network_Task, {"data_kind": "pubmed"}
    if name == "citationnetwork":
        return Citation_Network_Task, {}
    if name == "ppi":
        return PPI_Task, {}
    if name == "varmisuse":
        return VarMisuse_Task, {}

    raise ValueError("Unknown task type '%s'" % name)


def name_to_model_class(name: str) -> Tuple[Type[Sparse_Graph_Model], Dict[str, Any]]:
    name = name.lower()
    if name in ["ggnn", "ggnn_model"]:
        return GGNN_Model, {}
    if name in ["gnn_edge_mlp", "gnn-edge-mlp", "gnn_edge_mlp_model"]:
        return GNN_Edge_MLP_Model, {}
    if name in ["gnn_edge_mlp0", "gnn-edge-mlp0", "gnn_edge_mlp0_model"]:
        return GNN_Edge_MLP_Model, {'num_edge_hidden_layers': 0}
    if name in ["gnn_edge_mlp1", "gnn-edge-mlp1", "gnn_edge_mlp1_model"]:
        return GNN_Edge_MLP_Model, {'num_edge_hidden_layers': 1}
    if name in ["gnn_edge_mlp", "gnn-edge-mlp"]:
        return GNN_Edge_MLP_Model, {}
    if name in ["gnn_film", "gnn-film", "gnn_film_model"]:
        return GNN_FiLM_Model, {}
    if name in ["rgat", "rgat_model"]:
        return RGAT_Model, {}
    if name in ["rgcn", "rgcn_model"]:
        return RGCN_Model, {}
    if name in ["rgdcn", "rgdcn_model"]:
        return RGDCN_Model, {}
    if name in ["rgin", "rgin_model"]:
        return RGIN_Model, {}

    raise ValueError("Unknown model type '%s'" % name)


def restore(saved_model_path: str, result_dir: str, run_id: str = None) -> Sparse_Graph_Model:
    print("Loading model from file %s." % saved_model_path)
    with open(saved_model_path, 'rb') as in_file:
        data_to_load = pickle.load(in_file)

    model_cls, _ = name_to_model_class(data_to_load['model_class'])
    task_cls, additional_task_params = name_to_task_class(data_to_load['task_class'])

    if run_id is None:
        run_id = "_".join([task_cls.name(), model_cls.name(data_to_load['model_params']), time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

    task = task_cls(data_to_load['task_params'])
    task.restore_from_metadata(data_to_load['task_metadata'])

    model = model_cls(data_to_load['model_params'], task, run_id, result_dir)
    model.load_weights(data_to_load['weights'])

    model.log_line("Loaded model from snapshot %s." % saved_model_path)

    return model
