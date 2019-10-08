from typing import Dict, Any, List

import tensorflow as tf

from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_ggnn_layer


class GGNN_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'hidden_size': 128,
            'graph_rnn_cell': 'GRU',  # RNN, GRU, or LSTM
            'graph_activation_function': "tanh",
            "message_aggregation_function": "sum",
            'graph_layer_input_dropout_keep_prob': 1.0,
            'graph_dense_between_every_num_gnn_layers': 10000,
            'graph_residual_connection_every_num_layers': 10000,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "GGNN"

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_ggnn_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            gated_unit_type=self.params['graph_rnn_cell'],
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
        )
