from typing import Dict, Any, List

import tensorflow as tf

from .sparse_graph_model import Sparse_Graph_Model
from tasks import Sparse_Graph_Task
from gnns import sparse_gnn_edge_mlp_layer


class GNN_Edge_MLP_Model(Sparse_Graph_Model):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'max_nodes_in_batch': 25000,
            'hidden_size': 128,
            "graph_activation_function": "gelu",
            "message_aggregation_function": "sum",
            'graph_inter_layer_norm': True,
            'use_target_state_as_input': True,
            'num_edge_hidden_layers': 1,
        })
        return params

    @staticmethod
    def name(params: Dict[str, Any]) -> str:
        return "GNN-Edge-MLP%i" % (params['num_edge_hidden_layers'])

    def __init__(self, params: Dict[str, Any], task: Sparse_Graph_Task, run_id: str, result_dir: str) -> None:
        super().__init__(params, task, run_id, result_dir)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int,
                         ) -> tf.Tensor:
        return sparse_gnn_edge_mlp_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
            use_target_state_as_input=self.params['use_target_state_as_input'],
            num_edge_hidden_layers=self.params['num_edge_hidden_layers'],
        )
