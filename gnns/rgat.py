from typing import List, Optional

import tensorflow as tf
from dpu_utils.tfutils import unsorted_segment_log_softmax

from utils import get_activation


def sparse_rgat_layer(node_embeddings: tf.Tensor,
                      adjacency_lists: List[tf.Tensor],
                      state_dim: Optional[int],
                      num_heads: int = 4,
                      num_timesteps: int = 1,
                      activation_function: Optional[str] = "tanh"
                      ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing using attention. This generalises
    the original GAT model (Velickovic et al., https://arxiv.org/pdf/1710.10903.pdf)
    to multiple edge types by using different weights for different edge types.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    In the setting for a single attention head, we compute new states as follows:
        h^t_{v, \ell} := W_\ell h^t_v
        e_{u, \ell, v} := LeakyReLU(\alpha_\ell^T * concat(h^t_{u, \ell}, h^t_{v, \ell}))
        a_v := softmax_{\ell, u with (u, v) \in A_\ell}(e_{u, \ell, v})
        h^{t+1}_v := \sigma(\sum_{ell, (u, v) \in A_\ell}
                                a_v_{u, \ell} * h^_{u, \ell})
    The learnable parameters of this are the W_\ell \in R^{D, D} and \alpha_\ell \in R^{2*D}.

    In practice, we use K attention heads, computing separate, partial new states h^{t+1}_{v,k}
    and compute h^{t+1}_v as the concatentation of the partial states.
    For this, we reduce the shape of W_\ell to R^{D, D/K} and \alpha_\ell to R^{2*D/K}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * D: state dimension
    * K: number of attention heads
    * L: number of different edge types
    * E: number of edges of a given edge type

    Arguments:
        node_embeddings: float32 tensor of shape [V, D], the original representation of
            each node in the graph.
        adjacency_lists: List of L adjacency lists, represented as int32 tensors of shape
            [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
            of type l connects node v to node u.
        state_dim: Optional size of output dimension of the GNN layer. If not set, defaults
            to D, the dimensionality of the input. If different from the input dimension,
            parameter num_timesteps has to be 1.
        num_heads: Number of attention heads to use.
        num_timesteps: Number of repeated applications of this message passing layer.
        activation_function: Type of activation function used.

    Returns:
        float32 tensor of shape [V, state_dim]
    """
    num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]
    if state_dim is None:
        state_dim = tf.shape(node_embeddings, out_type=tf.int32)[1]
    per_head_dim = state_dim // num_heads

    # === Prepare things we need across all timesteps:
    activation_fn = get_activation(activation_function)
    edge_type_to_state_transformation_layers = []  # Layers to compute the message from a source state
    edge_type_to_attention_parameters = []  # Parameters for the attention mechanism
    edge_type_to_message_targets = []  # List of tensors of message targets
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        edge_type_to_state_transformation_layers.append(
            tf.keras.layers.Dense(units=state_dim,
                                  use_bias=False,
                                  activation=None,
                                  name="Edge_%i_Weight" % edge_type_idx))
        edge_type_to_attention_parameters.append(
            tf.get_variable(shape=(2 * state_dim),
                            name="Edge_%i_Attention_Parameters" % edge_type_idx))
        edge_type_to_message_targets.append(adjacency_list_for_edge_type[:, 1])

    # Let M be the number of messages (sum of all E):
    message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

    cur_node_states = node_embeddings
    for _ in range(num_timesteps):
        edge_type_to_per_head_messages = []  # type: List[tf.Tensor]  # list of lists of tensors of messages of shape [E, K, D/K]
        edge_type_to_per_head_attention_coefficients = []  # type: List[tf.Tensor]  # list of lists of tensors of shape [E, K]

        # Collect incoming messages per edge type
        # Note:
        #  We compute the state transformations (to make use of the wider, faster matrix multiplication),
        #  and then split into the individual attention heads via some reshapes:
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]

            transformed_states = \
                edge_type_to_state_transformation_layers[edge_type_idx](cur_node_states)  # Shape [V, D]

            edge_transformed_source_states = \
                tf.nn.embedding_lookup(params=transformed_states, ids=edge_sources)  # Shape [E, D]
            edge_transformed_target_states = \
                tf.nn.embedding_lookup(params=transformed_states, ids=edge_targets)  # Shape [E, D]

            per_edge_per_head_transformed_source_states = \
                tf.reshape(edge_transformed_source_states, shape=(-1, num_heads, per_head_dim))

            per_edge_per_head_transformed_states = \
                tf.concat([per_edge_per_head_transformed_source_states,
                           tf.reshape(edge_transformed_target_states, shape=(-1, num_heads, per_head_dim))],
                          axis=-1)  # Shape [E, K, 2*D/K]
            per_head_attention_pars = tf.reshape(edge_type_to_attention_parameters[edge_type_idx],
                                                 shape=(num_heads, 2 * per_head_dim))  # Shape [K, 2*D/K]
            per_edge_per_head_attention_coefficients = \
                tf.nn.leaky_relu(tf.einsum('vki,ki->vk',
                                           per_edge_per_head_transformed_states,
                                           per_head_attention_pars))  # Shape [E, K]

            edge_type_to_per_head_messages.append(per_edge_per_head_transformed_source_states)
            edge_type_to_per_head_attention_coefficients.append(per_edge_per_head_attention_coefficients)

        per_head_messages = tf.concat(edge_type_to_per_head_messages, axis=0)
        per_head_attention_coefficients = tf.concat(edge_type_to_per_head_attention_coefficients, axis=0)

        head_to_aggregated_messages = []  # list of tensors of shape [V, D/K]
        for head_idx in range(num_heads):
            # Compute the softmax over all the attention coefficients for all messages going to this state:
            attention_coefficients = tf.concat(per_head_attention_coefficients[:, head_idx], axis=0)  # Shape [M]
            attention_values = \
                tf.exp(unsorted_segment_log_softmax(logits=attention_coefficients,
                                                    segment_ids=message_targets,
                                                    num_segments=num_nodes))  # Shape [M]
            messages = per_head_messages[:, head_idx, :]  # Shape [M, D/K]
            # Compute weighted sum per target node for this head:
            head_to_aggregated_messages.append(
                tf.unsorted_segment_sum(data=tf.expand_dims(attention_values, -1) * messages,
                                        segment_ids=message_targets,
                                        num_segments=num_nodes))

        new_node_states = activation_fn(tf.concat(head_to_aggregated_messages, axis=-1))
        cur_node_states = new_node_states

    return cur_node_states
