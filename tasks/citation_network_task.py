from collections import namedtuple
from typing import Any, Dict, List, Iterable, Iterator

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath, LocalPath

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils.citation_network_utils import load_data, preprocess_features


CitationData = namedtuple('CitationData', ['adj_lists', 'num_incoming_edges', 'features', 'labels', 'mask'])


class Citation_Network_Task(Sparse_Graph_Task):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'add_self_loop_edges': True,
            'use_graph': True,
            'activation_function': "tanh",
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def name() -> str:
        return "CitationNetwork"

    @staticmethod
    def default_data_path() -> str:
        return "data/citation-networks"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 2
        self.__initial_node_feature_size = 0
        self.__num_output_classes = 0

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata['initial_node_feature_size'] = self.__initial_node_feature_size
        metadata['num_output_classes'] = self.__num_output_classes
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)
        self.__initial_node_feature_size = metadata['initial_node_feature_size']
        self.__num_output_classes = metadata['num_output_classes']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__initial_node_feature_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        train_data, valid_data, _ = self.__load_data(path)
        self._loaded_data[DataFold.TRAIN] = train_data
        self._loaded_data[DataFold.VALIDATION] = valid_data

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        _, _, test_data = self.__load_data(path)
        return test_data

    def __load_data(self, data_directory: RichPath):
        assert isinstance(data_directory, LocalPath), "CitationNetworkTask can only handle local data"
        data_path = data_directory.path
        print(" Loading CitationNetwork data from %s." % (data_path,))
        (adj_list, features, train_labels, valid_labels, test_labels, train_mask, valid_mask, test_mask) = \
            load_data(data_path, self.params['data_kind'])
        self.__initial_node_feature_size = features.shape[1]
        self.__num_output_classes = train_labels.shape[1]
        features = preprocess_features(features)

        train_data = \
            [self.__preprocess_data(adj_list, features, np.argmax(train_labels, axis=1), train_mask)]
        valid_data = \
            [self.__preprocess_data(adj_list, features, np.argmax(valid_labels, axis=1), valid_mask)]
        test_data = \
            [self.__preprocess_data(adj_list, features, np.argmax(test_labels, axis=1), test_mask)]
        return train_data, valid_data, test_data

    def __preprocess_data(self, adj_list: Dict[int, List[int]], features, labels, mask) -> CitationData:
        flat_adj_list = []
        self_loop_adj_list = []
        num_incoming_edges = np.zeros(shape=[len(adj_list)], dtype=np.int32)
        for node, neighbours in adj_list.items():
            for neighbour in neighbours:
                flat_adj_list.append((node, neighbour))
                flat_adj_list.append((neighbour, node))
                num_incoming_edges[neighbour] += 1
                num_incoming_edges[node] += 1
            self_loop_adj_list.append((node, node))

        # Prepend the self-loop information:
        num_incoming_edges = np.stack([np.ones_like(num_incoming_edges, dtype=np.int32),
                                       num_incoming_edges])  # Shape [2, V]
        return CitationData(adj_lists=[self_loop_adj_list, flat_adj_list],
                            num_incoming_edges=num_incoming_edges,
                            features=features,
                            labels=labels,
                            mask=mask)

    # -------------------- Model Construction --------------------
    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['labels'] = tf.placeholder(tf.int32, [None], name='labels')
        placeholders['mask'] = tf.placeholder(tf.float32, [None], name='mask')
        placeholders['out_layer_dropout_keep_prob'] =\
            tf.placeholder_with_default(input=tf.constant(1.0, dtype=tf.float32),
                                        shape=[],
                                        name='out_layer_dropout_keep_prob')

        final_node_representations = \
            tf.nn.dropout(model_ops['final_node_representations'],
                          rate=1.0 - placeholders['out_layer_dropout_keep_prob'])
        output_label_logits = \
            tf.keras.layers.Dense(units=self.__num_output_classes,
                                  use_bias=False,
                                  activation=None,
                                  name="OutputDenseLayer",
                                  )(final_node_representations)  # Shape [V, Classes]

        num_masked_preds = tf.reduce_sum(placeholders['mask'])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_label_logits,
                                                                labels=placeholders['labels'])
        total_loss = tf.reduce_sum(losses * placeholders['mask'])

        correct_preds = tf.equal(tf.argmax(output_label_logits, axis=1, output_type=tf.int32),
                                 placeholders['labels'])
        num_masked_correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32) * placeholders['mask'])
        accuracy = num_masked_correct / num_masked_preds
        tf.summary.scalar('accuracy', accuracy)

        model_ops['task_metrics'] = {
            'loss': total_loss / num_masked_preds,
            'total_loss': total_loss,
            'accuracy': accuracy,
        }

    # -------------------- Minibatching and training loop --------------------
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int) \
            -> Iterator[MinibatchData]:
        data = next(iter(data))  # type: CitationData
        if data_fold == DataFold.TRAIN:
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            out_layer_dropout_keep_prob = 1.0

        feed_dict = {
            model_placeholders['initial_node_features']: data.features,
            model_placeholders['adjacency_lists'][0]: data.adj_lists[0],
            model_placeholders['adjacency_lists'][1]: data.adj_lists[1],
            model_placeholders['type_to_num_incoming_edges']: data.num_incoming_edges,
            model_placeholders['num_graphs']: 1,
            model_placeholders['labels']: data.labels,
            model_placeholders['mask']: data.mask,
            model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
        }

        yield MinibatchData(feed_dict=feed_dict,
                            num_graphs=1,
                            num_nodes=data.features.shape[0],
                            num_edges=sum(len(adj_list) for adj_list in data.adj_lists))

    def early_stopping_metric(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> float:
        # Early stopping based on average loss:
        return np.sum([m['total_loss'] for m in task_metric_results]) / num_graphs

    def pretty_print_epoch_task_metrics(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> str:
        return "Acc: %.2f%%" % (task_metric_results[0]['accuracy'] * 100,)
