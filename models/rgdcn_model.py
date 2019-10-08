from typing import Dict, Any, List

import tensorflow as tf

from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_rgdcn_layer


class RGDCN_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'max_nodes_in_batch': 25000,
            'hidden_size': 128,
            'num_channels': 8,
            "use_full_state_for_channel_weights": False,
            "tie_channel_weights": False,
            "graph_activation_function": "ReLU",
            "message_aggregation_function": "sum",
            'graph_inter_layer_norm': True,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "RGDCN"

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        params['channel_dim'] = params['hidden_size'] // params['num_channels']
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_rgdcn_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            num_channels=self.params['num_channels'],
            channel_dim=self.params['channel_dim'],
            num_timesteps=num_timesteps,
            use_full_state_for_channel_weights=self.params['use_full_state_for_channel_weights'],
            tie_channel_weights=self.params['tie_channel_weights'],
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
        )
