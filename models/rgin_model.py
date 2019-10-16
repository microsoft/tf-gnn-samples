from typing import Dict, Any, List

import tensorflow as tf

from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_rgin_layer


class RGIN_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'hidden_size': 128,
            "graph_activation_function": "ReLU",
            'graph_layer_input_dropout_keep_prob': 1.0,
            'graph_dense_between_every_num_gnn_layers': 10000,
            'graph_residual_connection_every_num_layers': 10000,
            'graph_num_MLP_hidden_layers': 1,
            'graph_learn_epsilon': False,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "RGIN"

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_rgin_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            num_MLP_hidden_layers=self.params['graph_num_MLP_hidden_layers'],
            learn_epsilon=self.params['graph_learn_epsilon'],
        )
