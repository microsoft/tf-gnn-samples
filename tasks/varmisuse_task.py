import re
from collections import defaultdict
from multiprocessing import Process, Queue, cpu_count
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Iterator

import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath
from dpu_utils.codeutils import split_identifier_into_parts, get_language_keywords

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import BIG_NUMBER


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1
USES_SUBTOKEN_EDGE_NAME = "UsesSubtoken"
SELF_LOOP_EDGE_NAME = "SelfLoop"
BACKWARD_EDGE_TYPE_NAME_SUFFIX = "_Bkwd"
__PROGRAM_GRAPH_EDGES_TYPES = ["Child", "NextToken", "LastUse", "LastWrite", "LastLexicalUse", "ComputedFrom",
                               "GuardedByNegation", "GuardedBy", "FormalArgName", "ReturnsTo", USES_SUBTOKEN_EDGE_NAME]
__PROGRAM_GRAPH_EDGES_TYPES_WITH_BKWD = \
    __PROGRAM_GRAPH_EDGES_TYPES + [edge_type_name + BACKWARD_EDGE_TYPE_NAME_SUFFIX
                                   for edge_type_name in __PROGRAM_GRAPH_EDGES_TYPES]
PROGRAM_GRAPH_EDGES_TYPES_VOCAB = {edge_type_name: idx
                                   for idx, edge_type_name in enumerate(__PROGRAM_GRAPH_EDGES_TYPES_WITH_BKWD)}


class GraphSample(NamedTuple):
    adjacency_lists: List[np.ndarray]
    type_to_node_to_num_incoming_edges: np.ndarray
    unique_labels_as_characters: np.ndarray
    node_labels_to_unique_labels: np.ndarray
    slot_node_id: int
    variable_candidate_nodes: np.ndarray
    variable_candidate_nodes_mask: np.ndarray


def _add_per_subtoken_nodes(unsplittable_node_names: Set[str], graph_dict: Dict[str, Any]) -> None:
    graph_node_labels = graph_dict['NodeLabels']
    subtoken_to_using_nodes = defaultdict(set)

    max_used_node_id = 0
    for node_id, node_label in graph_node_labels.items():
        node_id = int(node_id)
        max_used_node_id = max(node_id, max_used_node_id)

        # Skip AST nodes and punctuation:
        if node_label in unsplittable_node_names:
            continue

        for subtoken in split_identifier_into_parts(node_label):
            if re.search('[a-zA-Z0-9]', subtoken):
                subtoken_to_using_nodes[subtoken].add(node_id)

    subtoken_node_id = max_used_node_id
    new_edges = []
    for subtoken, using_nodes in subtoken_to_using_nodes.items():
        subtoken_node_id += 1
        graph_node_labels[str(subtoken_node_id)] = subtoken
        new_edges.extend([(using_node_id, subtoken_node_id)
                          for using_node_id in using_nodes])

    graph_dict['Edges'][USES_SUBTOKEN_EDGE_NAME] = new_edges


def _load_single_sample(raw_sample: Dict[str, Any],
                        unsplittable_node_names: Set[str],
                        graph_node_label_max_num_chars: int,
                        max_variable_candidates: int = 5,
                        add_self_loop_edges: bool = False):
    _add_per_subtoken_nodes(unsplittable_node_names, raw_sample['ContextGraph'])
    num_nodes = len(raw_sample['ContextGraph']['NodeLabels'])

    node_label_chars = np.zeros(shape=(num_nodes, graph_node_label_max_num_chars),
                                dtype=np.uint8)
    for (node, label) in raw_sample['ContextGraph']['NodeLabels'].items():
        for (char_idx, label_char) in enumerate(label[:graph_node_label_max_num_chars].lower()):
            node_label_chars[int(node), char_idx] = ALPHABET_DICT.get(label_char, 1)
    node_label_chars_unique, node_label_chars_indices = np.unique(node_label_chars,
                                                                  axis=0,
                                                                  return_inverse=True)

    # Split edges according to edge_type and count their numbers:
    num_edge_types = len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)
    adjacency_lists = [np.zeros((0, 2), dtype=np.int32) for _ in range(num_edge_types)]
    num_incoming_edges_per_type = np.zeros((num_edge_types, num_nodes), dtype=np.uint16)
    raw_edges = raw_sample['ContextGraph']['Edges']
    for e_type, e_type_edges in raw_edges.items():
        if len(e_type_edges) > 0:
            e_type_bkwd = e_type + BACKWARD_EDGE_TYPE_NAME_SUFFIX
            e_type_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[e_type]
            e_type_bkwd_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[e_type_bkwd]

            fwd_edges = np.array(e_type_edges, dtype=np.int32)
            bkwd_edges = np.flip(fwd_edges, axis=1)

            adjacency_lists[e_type_idx] = fwd_edges
            adjacency_lists[e_type_bkwd_idx] = bkwd_edges
            num_incoming_edges_per_type[e_type_idx, :] = \
                np.bincount(adjacency_lists[e_type_idx][:, 1], minlength=num_nodes)
            num_incoming_edges_per_type[e_type_bkwd_idx, :] = \
                np.bincount(adjacency_lists[e_type_bkwd_idx][:, 1], minlength=num_nodes)

    if add_self_loop_edges:
        self_loop_edge_type_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[SELF_LOOP_EDGE_NAME]
        adjacency_lists[self_loop_edge_type_idx] = \
            np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=1)
        num_incoming_edges_per_type[self_loop_edge_type_idx, :] = \
            np.ones(shape=(num_nodes,))

    # VarMisuse-specific things: Reorder symbol candidates so that correct one is first.
    correct_candidate_id = None
    distractor_candidate_ids = []  # type: List[int]
    for candidate in raw_sample['SymbolCandidates']:
        if candidate['IsCorrect']:
            correct_candidate_id = candidate['SymbolDummyNode']
        else:
            distractor_candidate_ids.append(candidate['SymbolDummyNode'])
    assert correct_candidate_id is not None
    candidate_node_ids = [correct_candidate_id] + distractor_candidate_ids[:max_variable_candidates - 1]
    # Pad symbol candidates up to max_variable_candidates:
    num_scope_padding = max_variable_candidates - len(candidate_node_ids)
    candidate_node_ids_mask = [True] * len(candidate_node_ids) + [False] * num_scope_padding
    candidate_node_ids = candidate_node_ids + [0] * num_scope_padding

    return GraphSample(adjacency_lists=adjacency_lists,
                       type_to_node_to_num_incoming_edges=num_incoming_edges_per_type,
                       unique_labels_as_characters=node_label_chars_unique,
                       node_labels_to_unique_labels=node_label_chars_indices,
                       slot_node_id=raw_sample['SlotDummyNode'],
                       variable_candidate_nodes=np.array(candidate_node_ids),
                       variable_candidate_nodes_mask=np.array(candidate_node_ids_mask),
                       )


def _data_loading_worker(path_queue: Queue,
                         result_queue: Queue,
                         unsplittable_node_names: Set[str],
                         graph_node_label_max_num_chars: int,
                         max_variable_candidates: int,
                         add_self_loop_edges: bool,
                         ) -> None:
    while True:
        next_path = path_queue.get()
        if next_path is None:  # Our signal that all files have been processed
            path_queue.put(None)  # Signal to the other workers
            result_queue.put(None)  # Signal to the controller that we are done
            break

        # Read the file and push examples out as soon as we get them:
        for raw_sample in next_path.read_by_file_suffix():
            result_queue.put(_load_single_sample(raw_sample,
                                                 unsplittable_node_names,
                                                 graph_node_label_max_num_chars,
                                                 max_variable_candidates,
                                                 add_self_loop_edges,
                                                 ))


def _load_data(paths: List[RichPath],
               unsplittable_node_names: Set[str],
               graph_node_label_max_num_chars: int,
               max_variable_candidates: int,
               add_self_loop_edges: bool,
               no_parallel: bool = False,
               ) -> Iterable[GraphSample]:
    if no_parallel:
        for path in paths:
            for raw_sample in path.read_by_file_suffix():
                yield _load_single_sample(raw_sample,
                                          unsplittable_node_names,
                                          graph_node_label_max_num_chars,
                                          max_variable_candidates,
                                          add_self_loop_edges,
                                          )

    path_queue = Queue(maxsize=len(paths) + 1)
    result_queue = Queue()

    # Set up list of work to do:
    for path in paths:
        path_queue.put(path)
    path_queue.put(None)  # Signal for the end of the queue

    # Set up workers:
    workers = []
    for _ in range(cpu_count()):
        workers.append(Process(target=_data_loading_worker,
                               args=(path_queue,
                                     result_queue,
                                     unsplittable_node_names,
                                     graph_node_label_max_num_chars,
                                     max_variable_candidates,
                                     add_self_loop_edges,
                                     )))
        workers[-1].start()

    # Consume the data:
    num_workers_terminated = 0
    while num_workers_terminated < len(workers):
        parsed_sample = result_queue.get()
        if parsed_sample is None:
            num_workers_terminated += 1  # Worker signaled that it's done
        else:
            yield parsed_sample

    # Clean up the workers:
    for worker in workers:
        worker.join()


class VarMisuse_Task(Sparse_Graph_Task):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'max_variable_candidates': 5,
            'graph_node_label_max_num_chars': 19,
            'graph_node_label_representation_size': 64,
            'slot_score_via_linear_layer': True,
            'loss_function': 'max-likelihood',  # max-likelihood or max-margin
            'max-margin_loss_margin': 0.2,
            'out_layer_dropout_rate': 0.2,
            'add_self_loop_edges': False,
            # 'max_num_data_files': 3,
        })
        return params

    @staticmethod
    def name() -> str:
        return "VarMisuse"

    @staticmethod
    def default_data_path() -> str:
        return "data/varmisuse"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # If required, add the self-loop edge type to the vocab:
        if params.get('add_self_loop_edges'):
            if SELF_LOOP_EDGE_NAME not in PROGRAM_GRAPH_EDGES_TYPES_VOCAB:
                PROGRAM_GRAPH_EDGES_TYPES_VOCAB[SELF_LOOP_EDGE_NAME] = \
                    len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)

    @property
    def num_edge_types(self) -> int:
        return len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)

    @property
    def initial_node_feature_size(self) -> int:
        return self.params['graph_node_label_representation_size']

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        # Note that as __load_data produces a generator, we explicitly force loading
        # (and caching) here:
        self._loaded_data[DataFold.TRAIN] = \
            list(self.__load_data(path.join("graphs-train"), DataFold.TRAIN))
        self._loaded_data[DataFold.VALIDATION] = \
            list(self.__load_data(path.join("graphs-valid"), DataFold.VALIDATION))

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        if path.path == self.default_data_path():
            path = path.join("graphs-test")
        return iter(self.__load_data(path, DataFold.TEST))

    def __load_data(self, data_dir: RichPath, data_fold: DataFold) -> Iterator[GraphSample]:
        all_data_files = data_dir.iterate_filtered_files_in_dir("*.gz")

        max_num_files = self.params.get('max_num_data_files', None)
        if max_num_files is not None:
            all_data_files = sorted(all_data_files)[:max_num_files]
        else:
            all_data_files = list(all_data_files)
        print(" Loading VarMisuse data from %s [%i data files]." % (data_dir, len(all_data_files)))

        unsplittable_keywords = get_language_keywords('csharp')
        return _load_data(all_data_files,
                          unsplittable_keywords,
                          self.params['graph_node_label_max_num_chars'],
                          self.params['max_variable_candidates'],
                          self.params['add_self_loop_edges'])

    # -------------------- Model Construction --------------------
    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        node_label_char_length = self.params['graph_node_label_max_num_chars']
        placeholders['unique_labels_as_characters'] = \
            tf.placeholder(dtype=tf.int32, shape=[None, node_label_char_length], name='unique_labels_as_characters')
        placeholders['node_labels_to_unique_labels'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='node_labels_to_unique_labels')
        placeholders['adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        model_ops['initial_node_features'] = \
            self.__get_node_label_charcnn_embeddings(placeholders['unique_labels_as_characters'],
                                                     placeholders['node_labels_to_unique_labels'])
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']

    def __get_node_label_charcnn_embeddings(self,
                                            unique_labels_as_characters: tf.Tensor,
                                            node_labels_to_unique_labels: tf.Tensor,
                                            ) -> tf.Tensor:
        """
        Compute representation of node labels using a 2-layer character CNN.

        Args:
            unique_labels_as_characters: int32 tensor of shape [U, C]
                representing the unique (node) labels occurring in a
                batch, where U is the number of such labels and C the
                maximal number of characters.
            node_labels_to_unique_labels: int32 tensor of shape [V],
                mapping each node in the batch to one of the unique
                labels.

        Returns:
            float32 tensor of shape [V, D] representing embedded node
            label information about each node.
        """
        label_embedding_size = self.params['graph_node_label_representation_size']  # D
        # U ~ num unique labels
        # C ~ num characters (self.params['graph_node_label_max_num_chars'])
        # A ~ num characters in alphabet
        unique_label_chars_one_hot = tf.one_hot(indices=unique_labels_as_characters,
                                                depth=len(ALPHABET),
                                                axis=-1)  # Shape: [U, C, A]

        # Choose kernel sizes such that there is a single value at the end:
        char_conv_l1_kernel_size = 5
        char_conv_l2_kernel_size = \
            self.params['graph_node_label_max_num_chars'] - 2 * (char_conv_l1_kernel_size - 1)

        char_conv_l1 = \
            tf.keras.layers.Conv1D(filters=16,
                                   kernel_size=char_conv_l1_kernel_size,
                                   activation=tf.nn.leaky_relu,
                                   )(unique_label_chars_one_hot)  # Shape: [U, C - (char_conv_l1_kernel_size - 1), 16]
        char_pool_l1 = \
            tf.keras.layers.MaxPool1D(pool_size=char_conv_l1_kernel_size,
                                      strides=1,
                                      )(inputs=char_conv_l1)      # Shape: [U, C - 2*(char_conv_l1_kernel_size - 1), 16]
        char_conv_l2 = \
            tf.keras.layers.Conv1D(filters=label_embedding_size,
                                   kernel_size=char_conv_l2_kernel_size,
                                   activation=tf.nn.leaky_relu,
                                   )(char_pool_l1)                # Shape: [U, 1, D]
        unique_label_representations = tf.squeeze(char_conv_l2, axis=1)  # Shape: [U, D]
        node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=node_labels_to_unique_labels)
        return node_label_representations

    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['slot_node_ids'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='slot_node_ids')
        placeholders['candidate_node_ids'] = \
            tf.placeholder(dtype=tf.int32, shape=[None, None], name='candidate_node_ids')
        placeholders['candidate_node_ids_mask'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, None], name='candidate_node_ids_mask')
        placeholders['out_layer_dropout_rate'] = \
            tf.placeholder_with_default(0.0, shape=[], name='out_layer_dropout_rate')

        final_node_repr_size = model_ops['final_node_representations'].shape.as_list()[-1]
        num_candidate_vars = self.params['max_variable_candidates']

        final_node_states = \
            tf.nn.dropout(model_ops['final_node_representations'],
                          rate=placeholders['out_layer_dropout_rate'])  # Shape: [V, D]

        # --- (1) Collect representation of slots and candidates:
        slot_representations = \
            tf.gather(params=final_node_states, indices=placeholders['slot_node_ids'])  # Shape: [G, D]
        # Make things fit into 1D gather:
        candidate_node_ids = tf.reshape(placeholders['candidate_node_ids'], shape=[-1])
        candidate_representations = \
            tf.gather(params=final_node_states, indices=candidate_node_ids)  # Shape: [G * Cands, D]
        candidate_representations = \
            tf.reshape(candidate_representations,
                       shape=[-1, num_candidate_vars, final_node_repr_size])  # Shape: [G, Cands, D]

        # --- (2) Compute match between final candidate representations and slot representation:
        slot_candidate_inner_product = \
            tf.einsum('sd,scd->sc', slot_representations, candidate_representations)  # Shape: [G, Cands]

        if self.params['slot_score_via_linear_layer']:
            repeated_slots = tf.tile(tf.expand_dims(slot_representations, axis=1),
                                     multiples=[1, num_candidate_vars, 1])  # Shape: [G, Cands, D]
            slot_cand_comb = tf.concat([candidate_representations,
                                        repeated_slots,
                                        tf.expand_dims(slot_candidate_inner_product, -1)],
                                       axis=2)  # Shape: [G, Cands, 2*D + 1]
            logits = tf.keras.layers.Dense(units=1,
                                           use_bias=False,
                                           activation=None,
                                           name='slot_score_linear_layer'
                                           )(slot_cand_comb)  # Shape: [G, Cands, 1]
            logits = tf.squeeze(logits, axis=-1)  # Shape: [G, Cands]
        else:
            logits = slot_candidate_inner_product

        logits += (1.0 - placeholders['candidate_node_ids_mask']) * -BIG_NUMBER

        # --- (3) Compute loss & metrics:
        loss_function = self.params['loss_function']
        # Note that by convention, the first candidate is always the correct one:
        correct_choices = tf.zeros([tf.shape(logits)[0]], dtype=tf.int32)
        if loss_function == 'max-likelihood':
            per_graph_loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_choices, logits=logits)
        elif loss_function == 'max-margin':
            log_probs = tf.nn.log_softmax(logits)
            correct_log_prob = log_probs[:, 0]
            max_wrong_log_prob = tf.reduce_max(log_probs[:, 1:], axis=1)
            per_graph_loss = \
                tf.nn.relu(max_wrong_log_prob - correct_log_prob + self.parameters['loss_margin'])
        else:
            raise Exception('Invalid loss function option: "%s"' % loss_function)

        prediction_is_correct = tf.equal(tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32),
                                         correct_choices)
        accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        model_ops['task_metrics'] = {
            'loss': tf.reduce_mean(per_graph_loss),
            'total_loss': tf.reduce_sum(per_graph_loss),
            'accuracy': accuracy,
            'num_correct_predictions': tf.reduce_sum(tf.cast(prediction_is_correct, tf.int32)),
        }

    # -------------------- Minibatching and training loop --------------------
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int) \
            -> Iterable[MinibatchData]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(data)

        if isinstance(data, Iterator):
            data_iter = data
        else:
            data_iter = iter(data)

        def init_raw_batch_data_holder() -> Dict[str, Any]:
            return {
                'adj_lists': [[] for _ in range(self.num_edge_types)],
                'type_to_num_in_edges': [],
                'uniq_labels_as_chars': [],
                'node_labels_to_uniq_labels': [],
                'slot_node_ids': [],
                'candidate_node_ids': [],
                'candidate_node_ids_mask': [],
                'num_graphs': 0,
                'node_offset': 0,
                'unique_label_offset': 0,
            }

        def finalise_batch_data(raw_batch_data: Dict[str, Any]) -> MinibatchData:
            batch_feed_dict = {
                model_placeholders['unique_labels_as_characters']: np.concatenate(raw_batch_data['uniq_labels_as_chars'], axis=0),
                model_placeholders['node_labels_to_unique_labels']: np.concatenate(raw_batch_data['node_labels_to_uniq_labels'], axis=0),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(raw_batch_data['type_to_num_in_edges'], axis=1),
                model_placeholders['slot_node_ids']: raw_batch_data['slot_node_ids'],
                model_placeholders['candidate_node_ids']: raw_batch_data['candidate_node_ids'],
                model_placeholders['candidate_node_ids_mask']: raw_batch_data['candidate_node_ids_mask'],
            }

            if data_fold == DataFold.TRAIN:
                model_placeholders['out_layer_dropout_rate'] = self.params['out_layer_dropout_rate']

            # Merge adjacency lists:
            num_edges = 0
            for i in range(self.num_edge_types):
                if len(raw_batch_data['adj_lists'][i]) > 0:
                    adj_list = np.concatenate(raw_batch_data['adj_lists'][i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                num_edges += adj_list.shape[0]
                batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list

            return MinibatchData(feed_dict=batch_feed_dict,
                                 num_graphs=raw_batch_data['num_graphs'],
                                 num_nodes=raw_batch_data['node_offset'],
                                 num_edges=num_edges)

        try:
            cur_batch_data = init_raw_batch_data_holder()
            while True:
                cur_graph = next(data_iter)
                # We pack until we cannot fit more graphs in the batch, yield, and continue:
                if cur_batch_data['node_offset'] + len(cur_graph.node_labels_to_unique_labels) >= max_nodes_per_batch:
                    yield finalise_batch_data(cur_batch_data)
                    cur_batch_data = init_raw_batch_data_holder()

                # Graph structure:
                for i in range(self.num_edge_types):
                    cur_batch_data['adj_lists'][i].append(cur_graph.adjacency_lists[i] + cur_batch_data['node_offset'])
                cur_batch_data['type_to_num_in_edges'].append(cur_graph.type_to_node_to_num_incoming_edges)

                # Node labels:
                cur_batch_data['uniq_labels_as_chars'].append(cur_graph.unique_labels_as_characters)
                cur_batch_data['node_labels_to_uniq_labels'].append(
                    cur_graph.node_labels_to_unique_labels + cur_batch_data['unique_label_offset'])
                cur_batch_data['unique_label_offset'] += cur_graph.unique_labels_as_characters.shape[0]

                # VarMisuse task bits:
                cur_batch_data['slot_node_ids'].append(cur_graph.slot_node_id + cur_batch_data['node_offset'])
                cur_batch_data['candidate_node_ids'].append(cur_graph.variable_candidate_nodes + cur_batch_data['node_offset'])
                cur_batch_data['candidate_node_ids_mask'].append(cur_graph.variable_candidate_nodes_mask)

                # Finally, update the offset we use to shift things during batch construction:
                cur_batch_data['num_graphs'] += 1
                cur_batch_data['node_offset'] += len(cur_graph.node_labels_to_unique_labels)
        except StopIteration:
            # Final batch, yield only if non-empty:
            if cur_batch_data['num_graphs'] > 0:
                yield finalise_batch_data(cur_batch_data)

    def early_stopping_metric(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> float:
        # Early stopping based on accuracy; as we are trying to minimize, negate it:
        acc = sum([m['num_correct_predictions'] for m in task_metric_results]) / float(num_graphs)
        return -acc

    def pretty_print_epoch_task_metrics(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> str:
        acc = sum([m['num_correct_predictions'] for m in task_metric_results]) / float(num_graphs)
        return "Accuracy: %.3f" % (acc,)
