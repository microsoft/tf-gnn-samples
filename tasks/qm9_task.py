from collections import namedtuple
from typing import Any, Dict, Tuple, List, Iterable

import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import MLP


GraphSample = namedtuple('GraphSample', ['adjacency_lists',
                                         'type_to_node_to_num_incoming_edges',
                                         'node_features',
                                         'target_values',
                                         ])


class QM9_Task(Sparse_Graph_Task):
    # These magic constants were obtained during dataset generation, as result of normalising
    # the values of target properties:
    CHEMICAL_ACC_NORMALISING_FACTORS = [0.066513725, 0.012235489, 0.071939046,
                                        0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926,
                                        0.00409976, 0.004527465, 0.012292586,
                                        0.037467458]

    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'task_ids': [0],

            'add_self_loop_edges': True,
            'tie_fwd_bkwd_edges': True,
            'use_graph': True,
            'activation_function': "tanh",
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def name() -> str:
        return "QM9"

    @staticmethod
    def default_data_path() -> str:
        return "data/qm9"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 0
        self.__annotation_size = 0

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata['num_edge_types'] = self.__num_edge_types
        metadata['annotation_size'] = self.__annotation_size
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)
        self.__num_edge_types = metadata['num_edge_types']
        self.__annotation_size = metadata['annotation_size']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__annotation_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("train.jsonl.gz"))
        self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("valid.jsonl.gz"))

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        if path.path == self.default_data_path():
            path = path.join("test.jsonl.gz")
        return self.__load_data(path)

    def __load_data(self, data_file: RichPath) -> List[GraphSample]:
        print(" Loading QM9 data from %s." % (data_file,))
        data = list(data_file.read_by_file_suffix())  # list() needed for .jsonl case, where .read*() is just a generator

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        if self.params['add_self_loop_edges']:
            num_fwd_edge_types += 1
        self.__num_edge_types = max(self.num_edge_types,
                                    num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd_edges'] else 2))
        self.__annotation_size = max(self.__annotation_size, len(data[0]["node_features"][0]))
        return self.__process_raw_graphs(data)

    def __process_raw_graphs(self, raw_data: Iterable[Any]) -> List[GraphSample]:
        processed_graphs = []
        for d in raw_data:
            (type_to_adjacency_list, type_to_num_incoming_edges) = \
                self.__graph_to_adjacency_lists(d['graph'], num_nodes=len(d["node_features"]))
            processed_graphs.append(
                GraphSample(adjacency_lists=type_to_adjacency_list,
                            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
                            node_features=d["node_features"],
                            target_values=[d["targets"][task_id][0] for task_id in self.params['task_ids']],
                            ))
        return processed_graphs

    def __graph_to_adjacency_lists(self, graph: Iterable[Tuple[int, int, int]], num_nodes: int) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        type_to_adj_list = [[] for _ in range(self.num_edge_types)]  # type: List[List[Tuple[int, int]]]
        type_to_num_incoming_edges = np.zeros(shape=(self.num_edge_types, num_nodes,))
        for src, e, dest in graph:
            if self.params['add_self_loop_edges']:
                fwd_edge_type = e  # 0 will be the self-loop type
            else:
                fwd_edge_type = e - 1  # Make edges start from 0
            type_to_adj_list[fwd_edge_type].append((src, dest))
            type_to_num_incoming_edges[fwd_edge_type, dest] += 1
            if self.params['tie_fwd_bkwd_edges']:
                type_to_adj_list[fwd_edge_type].append((dest, src))
                type_to_num_incoming_edges[fwd_edge_type, src] += 1

        if self.params['add_self_loop_edges']:
            # Add self-loop edges (idx 0, which isn't used in the data):
            for node in range(num_nodes):
                type_to_num_incoming_edges[0, node] = 1
                type_to_adj_list[0].append((node, node))

        type_to_adj_list = [np.array(sorted(adj_list), dtype=np.int32) if len(adj_list) > 0 else np.zeros(shape=(0, 2), dtype=np.int32)
                            for adj_list in type_to_adj_list]

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd_edges']):
            type_to_adj_list = type_to_adj_list[:self.num_edge_types // 2]  # We allocated too much earlier...
            for (edge_type, adj_list) in enumerate(type_to_adj_list):
                bwd_edge_type = self.num_edge_types // 2 + edge_type
                type_to_adj_list.append(np.array(sorted((y, x) for (x, y) in adj_list), dtype=np.int32))
                for (x, y) in adj_list:
                    type_to_num_incoming_edges[bwd_edge_type][y] += 1

        return type_to_adj_list, type_to_num_incoming_edges

    # -------------------- Model Construction --------------------
    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['graph_nodes_list'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='graph_nodes_list')
        placeholders['target_values'] = \
            tf.placeholder(dtype=tf.float32, shape=[len(self.params['task_ids']), None], name='target_values')
        placeholders['out_layer_dropout_keep_prob'] = \
            tf.placeholder(dtype=tf.float32, shape=[], name='out_layer_dropout_keep_prob')

        task_metrics = {}
        losses = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                regression_gate = \
                    MLP(out_size=1,
                        hidden_layers=[],
                        use_biases=True,
                        dropout_rate=1.0 - placeholders['out_layer_dropout_keep_prob'],
                        name="regression_gate")
                regression_transform = \
                    MLP(out_size=1,
                        hidden_layers=[],
                        use_biases=True,
                        dropout_rate=1.0 - placeholders['out_layer_dropout_keep_prob'],
                        name="regression")

                per_node_outputs = regression_transform(model_ops['final_node_representations'])
                gate_input = tf.concat([model_ops['final_node_representations'],
                                        model_ops['initial_node_features']],
                                       axis=-1)
                per_node_gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * per_node_outputs

                # Sum up all nodes per-graph
                per_graph_outputs = tf.unsorted_segment_sum(data=per_node_gated_outputs,
                                                            segment_ids=placeholders['graph_nodes_list'],
                                                            num_segments=placeholders['num_graphs'])
                per_graph_outputs = tf.squeeze(per_graph_outputs)  # [g]

                per_graph_errors = per_graph_outputs - placeholders['target_values'][internal_id, :]
                task_metrics['abs_err_task%i' % task_id] = tf.reduce_sum(tf.abs(per_graph_errors))
                tf.summary.scalar('mae_task%i' % task_id,
                                  task_metrics['abs_err_task%i' % task_id] / tf.cast(placeholders['num_graphs'], tf.float32))
                losses.append(tf.reduce_mean(0.5 * tf.square(per_graph_errors)))
        model_ops['task_metrics'] = task_metrics
        model_ops['task_metrics']['loss'] = tf.reduce_sum(losses)
        model_ops['task_metrics']['total_loss'] = model_ops['task_metrics']['loss'] * tf.cast(placeholders['num_graphs'], tf.float32)

    # -------------------- Minibatching and training loop --------------------
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int) \
            -> Iterable[MinibatchData]:
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
            batch_target_task_values = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]  # type: List[List[np.ndarray]]
            batch_type_to_num_incoming_edges = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs].node_features) < max_nodes_per_batch:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph.node_features)
                batch_node_features.extend(cur_graph.node_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph],
                                                      fill_value=num_graphs_in_batch,
                                                      dtype=np.int32))
                for i in range(self.num_edge_types):
                    batch_adjacency_lists[i].append(cur_graph.adjacency_lists[i] + node_offset)

                # Turn counters for incoming edges into np array:
                batch_type_to_num_incoming_edges.append(cur_graph.type_to_node_to_num_incoming_edges)
                batch_target_task_values.append(cur_graph.target_values)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                model_placeholders['initial_node_features']: np.array(batch_node_features),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(batch_type_to_num_incoming_edges, axis=1),
                model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                model_placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1, 0]),
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
        maes = {}
        for task_id in self.params['task_ids']:
            maes['mae_task%i' % task_id] = 0.
        fnum_graphs = float(num_graphs)
        for batch_task_metric_results in task_metric_results:
            for task_id in self.params['task_ids']:
                maes['mae_task%i' % task_id] += batch_task_metric_results['abs_err_task%i' % task_id] / fnum_graphs

        maes_str = " ".join("%i:%.5f" % (task_id, maes['mae_task%i' % task_id])
                            for task_id in self.params['task_ids'])
        # The following translates back from MAE on the property values normalised to the [0,1] range to the original scale:
        err_str = " ".join("%i:%.5f" % (task_id, maes['mae_task%i' % task_id] / self.CHEMICAL_ACC_NORMALISING_FACTORS[task_id])
                           for task_id in self.params['task_ids'])

        return "MAEs: %s | Error Ratios: %s" % (maes_str, err_str)
