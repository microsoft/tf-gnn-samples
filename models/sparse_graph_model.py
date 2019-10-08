import os
import pickle
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Iterable

import tensorflow as tf
import numpy as np
from dpu_utils.utils import ThreadedIterator, RichPath

from tasks import Sparse_Graph_Task, DataFold
from utils import get_activation


class Sparse_Graph_Model(ABC):
    """
    Abstract superclass of all graph models, defining core model functionality
    such as training loops, interaction with tasks, etc. Needs to be extended by
    concrete GNN implementations.
    """
    @classmethod
    def default_params(cls):
        return {
            'max_nodes_in_batch': 50000,

            'graph_num_layers': 8,
            'graph_num_timesteps_per_layer': 1,

            'graph_layer_input_dropout_keep_prob': 0.8,
            'graph_dense_between_every_num_gnn_layers': 1,
            'graph_model_activation_function': 'tanh',
            'graph_residual_connection_every_num_layers': 2,
            'graph_inter_layer_norm': False,

            'max_epochs': 10000,
            'patience': 25,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'learning_rate_decay': 0.98,
            'lr_for_num_graphs_per_batch': None,  # The LR is normalised so that we use it for exactly that number of graphs; no normalisation happens if the value is None
            'momentum': 0.85,
            'clamp_gradient_norm': 1.0,
            'random_seed': 0,
        }

    @staticmethod
    @abstractmethod
    def name(params: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def __init__(self,
                 params: Dict[str, Any],
                 task: Sparse_Graph_Task,
                 run_id: str,
                 result_dir: str) -> None:
        self.params = params
        self.task = task
        self.run_id = run_id
        self.result_dir = result_dir

        self.__placeholders = {}  # type: Dict[str, tf.Tensor]
        self.__ops = {}  # type: Dict[str, tf.Tensor]

        # Build the actual model
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.params['random_seed'])
            self.__make_model()

    @property
    def log_file(self):
        return os.path.join(self.result_dir, "%s.log" % self.run_id)

    @property
    def best_model_file(self):
        return os.path.join(self.result_dir, "%s_best_model.pickle" % self.run_id)

    # -------------------- Model Saving/Loading --------------------
    def initialize_model(self) -> None:
        with self.sess.graph.as_default():
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self.sess.run(init_op)

    def save_model(self, path: str) -> None:
        vars_to_retrieve = {}  # type: Dict[str, tf.Tensor]
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in vars_to_retrieve
            vars_to_retrieve[variable.name] = variable
        weights_to_save = self.sess.run(vars_to_retrieve)

        data_to_save = {
            "model_class": self.name(self.params),
            "task_class": self.task.name(),
            "model_params": self.params,
            "task_params": self.task.params,
            "task_metadata": self.task.get_metadata(),
            "weights": weights_to_save,
        }
        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        with self.graph.as_default():
            variables_to_initialize = []
            with tf.name_scope("restore"):
                restore_ops = []
                used_vars = set()
                for variable in self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    used_vars.add(variable.name)
                    if variable.name in weights:
                        restore_ops.append(variable.assign(weights[variable.name]))
                    else:
                        print('Freshly initializing %s since no saved value was found.' % variable.name)
                        variables_to_initialize.append(variable)
                for var_name in weights:
                    if var_name not in used_vars:
                        print('Saved weights for %s not used by model.' % var_name)
                restore_ops.append(tf.variables_initializer(variables_to_initialize))
                self.sess.run(restore_ops)

    # -------------------- Model Construction --------------------
    def __make_model(self):
        self.task.make_task_input_model(self.__placeholders, self.__ops)

        with tf.variable_scope("graph_model"):
            self.__placeholders['num_graphs'] = \
                tf.placeholder(dtype=tf.int64, shape=[], name='num_graphs')
            self.__placeholders['graph_layer_input_dropout_keep_prob'] = \
                tf.placeholder_with_default(1.0, shape=[], name='graph_layer_input_dropout_keep_prob')

            self.__build_graph_propagation_model()

        self.task.make_task_output_model(self.__placeholders, self.__ops)

        tf.summary.scalar('loss', self.__ops['task_metrics']['loss'])
        total_num_graphs_variable = \
            tf.get_variable(name='total_num_graphs',
                            shape=(),
                            dtype=tf.int64,
                            initializer=tf.zeros_initializer,
                            trainable=False)
        self.__ops['total_num_graphs'] = \
            tf.assign_add(total_num_graphs_variable, self.__placeholders['num_graphs'])
        self.__ops['tf_summaries'] = tf.summary.merge_all()

        # Print some stats:
        num_pars = 0
        for variable in tf.trainable_variables():
            num_pars += np.prod([dim.value for dim in variable.get_shape()])
        self.log_line("Model has %i parameters." % num_pars)

        # Now add the optimizer bits:
        self.__make_train_step()

    def __build_graph_propagation_model(self) -> tf.Tensor:
        h_dim = self.params['hidden_size']
        activation_fn = get_activation(self.params['graph_model_activation_function'])
        if self.task.initial_node_feature_size != self.params['hidden_size']:
            self.__ops['projected_node_features'] = \
                tf.keras.layers.Dense(units=h_dim,
                                      use_bias=False,
                                      activation=activation_fn,
                                      )(self.__ops['initial_node_features'])
        else:
            self.__ops['projected_node_features'] = self.__ops['initial_node_features']

        cur_node_representations = self.__ops['projected_node_features']
        last_residual_representations = tf.zeros_like(cur_node_representations)
        for layer_idx in range(self.params['graph_num_layers']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                cur_node_representations = \
                    tf.nn.dropout(cur_node_representations, rate=1.0 - self.__placeholders['graph_layer_input_dropout_keep_prob'])
                if layer_idx % self.params['graph_residual_connection_every_num_layers'] == 0:
                    t = cur_node_representations
                    if layer_idx > 0:
                        cur_node_representations += last_residual_representations
                        cur_node_representations /= 2
                    last_residual_representations = t
                cur_node_representations = \
                    self._apply_gnn_layer(
                        cur_node_representations,
                        self.__ops['adjacency_lists'],
                        self.__ops['type_to_num_incoming_edges'],
                        self.params['graph_num_timesteps_per_layer'])
                if self.params['graph_inter_layer_norm']:
                    cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
                if layer_idx % self.params['graph_dense_between_every_num_gnn_layers'] == 0:
                    cur_node_representations = \
                        tf.keras.layers.Dense(units=h_dim,
                                              use_bias=False,
                                              activation=activation_fn,
                                              name="Dense",
                                              )(cur_node_representations)

        self.__ops['final_node_representations'] = cur_node_representations

    @abstractmethod
    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        """
        Run a GNN layer on a graph.

        Arguments:
            node_features: float32 tensor of shape [V, D], where V is the number of nodes.
            adjacency_lists: list of L int32 tensors of shape [E, 2], where L is the number
                of edge types and E the number of edges of that type.
                Hence, adjacency_lists[l][e,:] == [u, v] means that u has an edge of type l
                to v.
            type_to_num_incoming_edges: int32 tensor of shape [L, V], where L is the number
                of edge types.
                type_to_num_incoming_edges[l, v] = k indicates that node v has k incoming
                edges of type l.
            num_timesteps: Number of propagation steps in to run in this GNN layer.
        """
        raise Exception("Models have to implement _apply_gnn_layer!")

    def __make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        learning_rate = self.params['learning_rate']

        lr_for_num_graphs_per_batch = self.params.get('lr_for_num_graphs_per_batch')
        if lr_for_num_graphs_per_batch is not None:
            # This ensures that the learning rate _per_ graph in the batch stays the same,
            # which can be important for tasks in which the loss is defined per-graph
            # (e.g., full graph regression tasks, or one-node-per-graph classification)
            lr_norm_factor = (tf.cast(self.__placeholders['num_graphs'], tf.float32)
                              / tf.constant(lr_for_num_graphs_per_batch, dtype=tf.float32))
            learning_rate *= lr_norm_factor

        optimizer_name = self.params['optimizer'].lower()
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=self.params['learning_rate_decay'],
                                                  momentum=self.params['momentum'])
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception('Unknown optimizer "%s".' % (self.params['optimizer']))

        grads_and_vars = optimizer.compute_gradients(self.__ops['task_metrics']['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.__ops['train_step'] = optimizer.apply_gradients(clipped_grads)

    # -------------------- Training Loop --------------------
    def __run_epoch(self,
                    epoch_name: str,
                    data: Iterable[Any],
                    data_fold: DataFold,
                    quiet: Optional[bool] = False,
                    summary_writer: Optional[tf.summary.FileWriter] = None) \
            -> Tuple[float, List[Dict[str, Any]], int, float, float, float]:
        batch_iterator = self.task.make_minibatch_iterator(
            data, data_fold, self.__placeholders, self.params['max_nodes_in_batch'])
        batch_iterator = ThreadedIterator(batch_iterator, max_queue_size=5)
        task_metric_results = []
        start_time = time.time()
        processed_graphs, processed_nodes, processed_edges = 0, 0, 0
        epoch_loss = 0.0
        for step, batch_data in enumerate(batch_iterator):
            if data_fold == DataFold.TRAIN:
                batch_data.feed_dict[self.__placeholders['graph_layer_input_dropout_keep_prob']] = \
                    self.params['graph_layer_input_dropout_keep_prob']
            batch_data.feed_dict[self.__placeholders['num_graphs']] = batch_data.num_graphs
            # Collect some statistics:
            processed_graphs += batch_data.num_graphs
            processed_nodes += batch_data.num_nodes
            processed_edges += batch_data.num_edges

            fetch_dict = {'task_metrics': self.__ops['task_metrics']}
            if summary_writer:
                fetch_dict['tf_summaries'] = self.__ops['tf_summaries']
                fetch_dict['total_num_graphs'] = self.__ops['total_num_graphs']
            if data_fold == DataFold.TRAIN:
                fetch_dict['train_step'] = self.__ops['train_step']
            fetch_results = self.sess.run(fetch_dict, feed_dict=batch_data.feed_dict)
            epoch_loss += fetch_results['task_metrics']['loss'] * batch_data.num_graphs
            task_metric_results.append(fetch_results['task_metrics'])

            if not quiet:
                print("Running %s, batch %i (has %i graphs). Loss so far: %.4f"
                      % (epoch_name, step, batch_data.num_graphs, epoch_loss / processed_graphs),
                      end='\r')
            if summary_writer:
                summary_writer.add_summary(fetch_results['tf_summaries'], fetch_results['total_num_graphs'])

        assert processed_graphs > 0, "Can't run epoch over empty dataset."

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_graphs
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time
        edges_per_sec = processed_edges / epoch_time
        return per_graph_loss, task_metric_results, processed_graphs, graphs_per_sec, nodes_per_sec, edges_per_sec

    def log_line(self, msg):
        with open(self.log_file, 'a') as log_fh:
            log_fh.write(msg + '\n')
        print(msg)

    def train(self, quiet: Optional[bool] = False, tf_summary_path: Optional[str] = None):
        total_time_start = time.time()
        with self.graph.as_default():
            if tf_summary_path is not None:
                os.makedirs(tf_summary_path, exist_ok=True)
                train_writer = tf.summary.FileWriter(os.path.join(tf_summary_path, "train"), graph=self.graph)
                valid_writer = tf.summary.FileWriter(os.path.join(tf_summary_path, "valid"))
            else:
                train_writer, valid_writer = None, None

            (best_valid_metric, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")
            for epoch in range(1, self.params['max_epochs'] + 1):
                self.log_line("== Epoch %i" % epoch)

                train_loss, train_task_metrics, train_num_graphs, train_graphs_p_s, train_nodes_p_s, train_edges_p_s = \
                    self.__run_epoch("epoch %i (training)" % epoch,
                                     self.task._loaded_data[DataFold.TRAIN],
                                     DataFold.TRAIN,
                                     quiet=quiet,
                                     summary_writer=train_writer)
                if not quiet:
                    print("\r\x1b[K", end='')
                self.log_line(" Train: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | edges/sec: %.0f"
                              % (train_loss,
                                 self.task.pretty_print_epoch_task_metrics(train_task_metrics, train_num_graphs),
                                 train_graphs_p_s, train_nodes_p_s, train_edges_p_s))

                valid_loss, valid_task_metrics, valid_num_graphs, valid_graphs_p_s, valid_nodes_p_s, valid_edges_p_s = \
                    self.__run_epoch("epoch %i (validation)" % epoch,
                                     self.task._loaded_data[DataFold.VALIDATION],
                                     DataFold.VALIDATION,
                                     quiet=quiet,
                                     summary_writer=valid_writer)
                if not quiet:
                    print("\r\x1b[K", end='')
                early_stopping_metric = self.task.early_stopping_metric(valid_task_metrics, valid_num_graphs)
                valid_metric_descr = \
                    self.task.pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs)
                self.log_line(" Valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | edges/sec: %.0f"
                              % (valid_loss, valid_metric_descr, valid_graphs_p_s, valid_nodes_p_s, valid_edges_p_s))

                if early_stopping_metric < best_valid_metric:
                    self.save_model(self.best_model_file)
                    self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')"
                                  % (early_stopping_metric, best_valid_metric, self.best_model_file))
                    best_valid_metric = early_stopping_metric
                    best_val_metric_epoch = epoch
                    best_val_metric_descr = valid_metric_descr
                elif epoch - best_val_metric_epoch >= self.params['patience']:
                    total_time = time.time() - total_time_start
                    self.log_line("Stopping training after %i epochs without improvement on validation loss." % self.params['patience'])
                    self.log_line("Training took %is. Best validation results: %s"
                                  % (total_time, best_val_metric_descr))
                    break

    def test(self, path: RichPath, quiet: Optional[bool] = False):
        with self.graph.as_default():
            self.log_line("== Running Test on %s ==" % (path,))
            data = self.task._loaded_data.get(DataFold.TEST)
            if data is None:
                data = self.task.load_eval_data_from_path(path)
            test_loss, test_task_metrics, test_num_graphs, _, _, _ = \
                self.__run_epoch("Test", data, DataFold.TEST, quiet=quiet)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line("Loss %.5f on %i graphs" % (test_loss, test_num_graphs))
            self.log_line("Metrics: %s" % self.task.pretty_print_epoch_task_metrics(test_task_metrics, test_num_graphs))
