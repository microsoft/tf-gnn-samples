from typing import Dict, Any, List

import tensorflow as tf

from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_gnn_film_layer


class GNN_FiLM_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            "hidden_size": 128,
            "graph_activation_function": "ReLU",
            "message_aggregation_function": "sum",
            "normalize_messages_by_num_incoming": False,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "GNN-FiLM"

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_gnn_film_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
            normalize_by_num_incoming=self.params["normalize_messages_by_num_incoming"],
        )
