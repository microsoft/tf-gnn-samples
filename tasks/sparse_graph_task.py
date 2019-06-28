from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional

import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class MinibatchData(NamedTuple):
    feed_dict: Dict[str, tf.Tensor]
    num_graphs: int
    num_nodes: int
    num_edges: int


class Sparse_Graph_Task(ABC):
    """
    Abstract superclass of all graph tasks, defining the interface used by the
    remainder of the code to interact with a task.
    """
    @classmethod
    def default_params(cls):
        return {}

    @staticmethod
    @abstractmethod
    def default_data_path() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self._loaded_data = {}  # type: Dict[DataFold, Any]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with all metadata that defines this task, for example parameters
            or vocabularies.
        """
        return {"params": self.params}

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set up task to match passed metadata, e.g., by using the passed vocabulary.
        The input can be expected to be an output of get_metadata from another run.
        """
        self.params = metadata["params"]

    @property
    @abstractmethod
    def num_edge_types(self) -> int:
        """
        Returns:
            Number of edge types used in the dataset.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def initial_node_feature_size(self) -> int:
        """
        Return:
            Size of the initial node representation.
        """
        raise NotImplementedError()

    @property
    def has_test_data(self) -> bool:
        return DataFold.TEST in self._loaded_data

    @abstractmethod
    def load_data(self, path: Optional[RichPath]) -> None:
        """
        Load data required to train on this task into memory.

        Arguments:
            path: Optional path to load from, if not specified, will use task-specific
                default under "./data/".
        """
        raise NotImplementedError()

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        """
        Load data from a given path for evaluation purposes.

        Arguments:
            path: Depending on the task a file or directory containing data to load.

        Returns:
            An iterator over graph samples, suitable for being passed into
            task.make_minibatch_iterator().
        """
        raise NotImplementedError()

    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        """
        Create a task-specific input model. The default implementation
        simply creates placeholders to feed the input in, but more advanced
        variants could include sub-networks determining node features,
        for example.

        This method cannot assume the placeholders or model_ops dictionaries
        to be pre-populated, and needs to add at least the following
        entries to model_ops:
         * 'initial_node_features': float32 tensor of shape [V, D], where V
           is the number of nodes and D is the initial hidden dimension
           (needs to match the value of task.initial_node_feature_size).
         * 'adjacency_lists': list of L int32 tensors of shape [E, 2], where
           L is the number of edge types and E the number of edges of that
           type.
           Hence, adjacency_lists[l][e,:] == [u, v] means that u has an edge
           of type l to v.
         * 'type_to_num_incoming_edges': int32 tensor of shape [L, V], where
           L is the number of edge types and V the number of nodes.
           type_to_num_incoming_edges[l, v] = k indicates that node v has k
           incoming edges of type l.

        Arguments:
            placeholders: Dictionary of placeholders used by the model, to
                be extended with task-specific placeholders.
            model_ops: Dictionary of named operations in the model, to
                be extended with task-specific operations.
        """
        placeholders['initial_node_features'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, self.initial_node_feature_size], name='initial_node_features')
        placeholders['adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        model_ops['initial_node_features'] = placeholders['initial_node_features']
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']

    @abstractmethod
    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        """
        Create task-specific output model. For this, additional placeholders
        can be created, but will need to be filled in the
        make_minibatch_iterator implementation.

        This method may assume existence of the placeholders and ops created in
        make_task_input_model and of the following:
            model_ops['final_node_representations']: a float32 tensor of shape
                [V, D], which holds the final node representations after the
                GNN layers.
            placeholders['num_graphs']: a int32 scalar holding the number of
                graphs in this batch.
        Order of nodes is preserved across all tensors.

        This method has to define model_ops['task_metrics'] to a dictionary,
        from which model_ops['task_metrics']['loss'] will be used for
        optimization. Other entries may hold additional metrics (accuracy,
        MAE, ...).

        Arguments:
            placeholders: Dictionary of placeholders used by the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific placeholders.
            model_ops: Dictionary of named operations in the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific operations.
        """
        raise NotImplementedError()

    @abstractmethod
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int,
                                ) -> Iterator[MinibatchData]:
        """
        Create minibatches for a sparse graph model, usually by flattening
        many smaller graphs into one large graphs of disconnected components.
        This should produce one epoch's worth of minibatches.

        Arguments:
            data: Data to iterate over, created by either load_data or
                load_eval_data_from_path.
            data_fold: Fold of the loaded data to iterate over.
            model_placeholders: The placeholders of the model that need to be
                filled with data. Aside from the placeholders introduced by the
                task in make_task_input_model and make_task_output_model.
            max_nodes_per_batch: Maximal number of nodes that can be packed
                into one batch.

        Returns:
            Iterator over MinibatchData values, which provide feed dicts
            as well as some batch statistics.
        """
        raise NotImplementedError()

    @abstractmethod
    def early_stopping_metric(self,
                              task_metric_results: List[Dict[str, np.ndarray]],
                              num_graphs: int,
                              ) -> float:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a metric that should go down (e.g., loss). This is used
        for early stopping of training.

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            Numeric value, where a lower value indicates more desirable results.
        """
        raise NotImplementedError()

    @abstractmethod
    def pretty_print_epoch_task_metrics(self,
                                        task_metric_results: List[Dict[str, np.ndarray]],
                                        num_graphs: int,
                                        ) -> str:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a human-readable result for the epoch (e.g., average
        accuracy).

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            String representation of the task-specific metrics for this epoch,
            e.g., mean absolute error for a regression task.
        """
        raise NotImplementedError()
