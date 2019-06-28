from collections import namedtuple
from typing import Any, Dict, Iterator, List, Iterable

import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import micro_f1


GraphSample = namedtuple('GraphSample', ['adjacency_lists',
                                         'type_to_node_to_num_incoming_edges',
                                         'node_features',
                                         'node_labels',
                                         ])


class PPI_Task(Sparse_Graph_Task):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'add_self_loop_edges': True,
            'tie_fwd_bkwd_edges': False,
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def name() -> str:
        return "PPI"

    @staticmethod
    def default_data_path() -> str:
        return "data/ppi"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 0
        self.__initial_node_feature_size = 0
        self.__num_labels = 0

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata['num_edge_types'] = self.__num_edge_types
        metadata['initial_node_feature_size'] = self.__initial_node_feature_size
        metadata['num_labels'] = self.__num_labels
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)
        self.__num_edge_types = metadata['num_edge_types']
        self.__initial_node_feature_size = metadata['initial_node_feature_size']
        self.__num_labels = metadata['num_labels']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__initial_node_feature_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        # Data in format as downloaded from https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip
        self._loaded_data[DataFold.TRAIN] = self.__load_data(path, DataFold.TRAIN)
        self._loaded_data[DataFold.VALIDATION] = self.__load_data(path, DataFold.VALIDATION)

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        return self.__load_data(path, DataFold.TEST)

    def __load_data(self, data_dir: RichPath, data_fold: DataFold) -> List[GraphSample]:
        if data_fold == DataFold.TRAIN:
            data_name = "train"
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
        elif data_fold == DataFold.TEST:
            data_name = "test"
        else:
            raise ValueError("Unknown data fold '%s'" % str(data_fold))
        print(" Loading PPI %s data from %s." % (data_name, data_dir))

        graph_json_data = data_dir.join("%s_graph.json" % data_name).read_by_file_suffix()
        node_to_features = data_dir.join("%s_feats.npy" % data_name).read_by_file_suffix()
        node_to_labels = data_dir.join("%s_labels.npy" % data_name).read_by_file_suffix()
        node_to_graph_id = data_dir.join("%s_graph_id.npy" % data_name).read_by_file_suffix()
        self.__initial_node_feature_size = node_to_features.shape[-1]
        self.__num_labels = node_to_labels.shape[-1]

        # We read in all the data in two steps:
        #  (1) Read features, labels and insert self-loop edges (edge type 0).
        #      Implicitly, this gives us the number of nodes per graph.
        #  (2) Read all edges, and shift them so that each graph starts with node 0.

        fwd_edge_type = 0
        self.__num_edge_types = 1
        if self.params['add_self_loop_edges']:
            self_loop_edge_type = self.__num_edge_types
            self.__num_edge_types += 1
        if not self.params['tie_fwd_bkwd_edges']:
            bkwd_edge_type = self.__num_edge_types
            self.__num_edge_types += 1

        graph_id_to_graph_data = {}  # type: Dict[int, GraphSample]
        graph_id_to_node_offset = {}
        num_total_nodes = node_to_features.shape[0]
        for node_id in range(num_total_nodes):
            graph_id = node_to_graph_id[node_id]
            # In case we are entering a new graph, note its ID, so that we can normalise everything to start at 0
            if graph_id not in graph_id_to_graph_data:
                graph_id_to_graph_data[graph_id] = \
                    GraphSample(adjacency_lists=[[] for _ in range(self.__num_edge_types)],
                                type_to_node_to_num_incoming_edges=[[] for _ in range(self.__num_edge_types)],
                                node_features=[],
                                node_labels=[])
                graph_id_to_node_offset[graph_id] = node_id
            cur_graph_data = graph_id_to_graph_data[graph_id]
            cur_graph_data.node_features.append(node_to_features[node_id])
            cur_graph_data.node_labels.append(node_to_labels[node_id])
            shifted_node_id = node_id - graph_id_to_node_offset[graph_id]
            if self.params['add_self_loop_edges']:
                cur_graph_data.adjacency_lists[self_loop_edge_type].append((shifted_node_id, shifted_node_id))
                cur_graph_data.type_to_node_to_num_incoming_edges[self_loop_edge_type].append(1)

        # Prepare reading of the edges by setting counters to 0:
        for graph_data in graph_id_to_graph_data.values():
            num_graph_nodes = len(graph_data.node_features)
            graph_data.type_to_node_to_num_incoming_edges[fwd_edge_type] = np.zeros([num_graph_nodes], np.int32)
            if not self.params['tie_fwd_bkwd_edges']:
                graph_data.type_to_node_to_num_incoming_edges[bkwd_edge_type] = np.zeros([num_graph_nodes], np.int32)

        for edge_info in graph_json_data['links']:
            src_node, tgt_node = edge_info['source'], edge_info['target']
            # First, shift node IDs so that each graph starts at node 0:
            graph_id = node_to_graph_id[src_node]
            graph_node_offset = graph_id_to_node_offset[graph_id]
            src_node, tgt_node = src_node - graph_node_offset, tgt_node - graph_node_offset

            cur_graph_data = graph_id_to_graph_data[graph_id]
            cur_graph_data.adjacency_lists[fwd_edge_type].append((src_node, tgt_node))
            cur_graph_data.type_to_node_to_num_incoming_edges[fwd_edge_type][tgt_node] += 1
            if not self.params['tie_fwd_bkwd_edges']:
                cur_graph_data.adjacency_lists[bkwd_edge_type].append((tgt_node, src_node))
                cur_graph_data.type_to_node_to_num_incoming_edges[bkwd_edge_type][src_node] += 1

        final_graphs = []
        for graph_data in graph_id_to_graph_data.values():
            # numpy-ize:
            adj_lists = []
            for edge_type_idx in range(self.__num_edge_types):
                adj_lists.append(np.array(graph_data.adjacency_lists[edge_type_idx]))
            final_graphs.append(
                GraphSample(adjacency_lists=adj_lists,
                            type_to_node_to_num_incoming_edges=np.array(graph_data.type_to_node_to_num_incoming_edges),
                            node_features=np.array(graph_data.node_features),
                            node_labels=np.array(graph_data.node_labels)))

        return final_graphs

    # -------------------- Model Construction --------------------
    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['graph_nodes_list'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='graph_nodes_list')
        placeholders['target_labels'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, self.__num_labels], name='target_labels')
        placeholders['out_layer_dropout_keep_prob'] = \
            tf.placeholder(dtype=tf.float32, shape=[], name='out_layer_dropout_keep_prob')

        per_node_logits = \
            tf.keras.layers.Dense(units=self.__num_labels,
                                  use_bias=True,
                                  )(model_ops['final_node_representations'])

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=per_node_logits,
                                                         labels=placeholders['target_labels'])
        total_loss = tf.reduce_sum(losses)

        # Compute loss as average per node (to account for changing number of nodes per batch):
        num_nodes_in_batch = tf.shape(placeholders['target_labels'])[0]

        f1_score = micro_f1(per_node_logits, placeholders['target_labels'])
        tf.summary.scalar("Micro F1", f1_score)
        model_ops['task_metrics'] = {
            'loss': total_loss / tf.cast(num_nodes_in_batch, tf.float32),
            'total_loss': total_loss,
            'f1_score': f1_score,
        }

    # -------------------- Minibatching and training loop --------------------
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int) \
            -> Iterator[MinibatchData]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(data)
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            out_layer_dropout_keep_prob = 1.0

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []  # type: List[np.ndarray]
            batch_node_labels = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]  # type: List[List[np.ndarray]]
            batch_type_to_num_incoming_edges = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs].node_features) < max_nodes_per_batch:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(data[num_graphs].node_features)
                batch_node_features.extend(cur_graph.node_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph],
                                                      fill_value=num_graphs_in_batch,
                                                      dtype=np.int32))
                for i in range(self.num_edge_types):
                    batch_adjacency_lists[i].append(cur_graph.adjacency_lists[i] + node_offset)
                batch_type_to_num_incoming_edges.append(cur_graph.type_to_node_to_num_incoming_edges)
                batch_node_labels.append(cur_graph.node_labels)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                model_placeholders['initial_node_features']: np.array(batch_node_features),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(batch_type_to_num_incoming_edges, axis=1),
                model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                model_placeholders['target_labels']: np.concatenate(batch_node_labels, axis=0),
                model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
            }

            # Merge adjacency lists:
            num_edges = 0
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                num_edges += adj_list.shape[0]
                batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list

            yield MinibatchData(feed_dict=batch_feed_dict,
                                num_graphs=num_graphs_in_batch,
                                num_nodes=node_offset,
                                num_edges=num_edges)

    def early_stopping_metric(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> float:
        # Early stopping based on average loss:
        return np.sum([m['total_loss'] for m in task_metric_results]) / num_graphs

    def pretty_print_epoch_task_metrics(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> str:
        avg_microf1 = np.average([m['f1_score'] for m in task_metric_results])
        return "Avg MicroF1: %.3f" % (avg_microf1,)
