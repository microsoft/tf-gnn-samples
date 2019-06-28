from typing import List, Optional

import tensorflow as tf

from utils import get_activation, get_aggregation_function, SMALL_NUMBER


def sparse_rgdcn_layer(node_embeddings: tf.Tensor,
                       adjacency_lists: List[tf.Tensor],
                       type_to_num_incoming_edges: tf.Tensor,
                       num_channels: int = 8,
                       channel_dim: int = 16,
                       num_timesteps: int = 1,
                       use_full_state_for_channel_weights: bool = False,
                       tie_channel_weights: bool = False,
                       activation_function: Optional[str] = "tanh",
                       message_aggregation_function: str = "sum",
                       normalize_by_num_incoming: bool = True,
                       ) -> tf.Tensor:
    """
    Compute new graph states by message passing using dynamic convolutions for edge kernels.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.
    We split each state h^t_v into C "channels" of dimension K, and use h^t_{v,c,:} to refer to
    the slice of the node state corresponding to the c-th channel.

    Four variants of the model are implemented:

    (1) Edge kernels computed from full target node state using weights shared across all channels:
        [use_full_state_for_channel_weights = True, tie_channel_weights = True]
          h^{t+1}_v := \sigma(Concat(\sum_\ell \sum_{(u, v) \in A_\ell} W^t_{\ell,v} * h^t_{u,c,:}
                                     | 1 <= c <= C))
          W^t_{\ell,v} := F_\ell * h^t_{v,:,:}
        The learnable parameters of this are the F_\ell \in R^{C*K, K*K}.

    (2) Edge kernels computed from full target node state using separate weights for each channel:
        [use_full_state_for_channel_weights = True, tie_channel_weights = False]
          h^{t+1}_v := \sigma(Concat(\sum_\ell \sum_{(u, v) \in A_\ell} \sigma(W^t_{\ell,v,c} * h^t_{u,c,:}
                                     | 1 <= c <= C)
          W^t_{\ell,v,c} := F_{\ell,c} * h^t_{v,:,:}
        The learnable parameters of this are the F_{\ell,c} \in R^{C*K, K*K}.

    (3) Edge kernels computed from corresponding channel of target node using weights shared across all channels:
        [use_full_state_for_channel_weights = False, tie_channel_weights = True]
          h^{t+1}_v := \sigma(Concat(\sum_\ell \sum_{(u, v) \in A_\ell} \sigma(W^t_{\ell,v,c} * h^t_{u,c,:}
                                     | 1 <= c <= C)
          W^t_{\ell,v,c} := F_{\ell} * h^t_{v,c,:}
        The learnable parameters of this are the F_\ell \in R^{K, K*K}.

    (4) Edge kernels computed from corresponding channel of target node using separate weights for each channel:
        [use_full_state_for_channel_weights = False, tie_channel_weights = False]
          h^{t+1}_v := \sigma(Concat(\sum_\ell \sum_{(u, v) \in A_\ell} W^t_{\ell,v,c} * h^t_{u,c,:}
                                     | 1 <= c <= C))
          W^t_{\ell,v,c} := F_{\ell,c} * h^t_{v,c,:}
        The learnable parameters of this are the F_{\ell,c} \in R^{K, K*K}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * C: number of "channels"
    * K: dimension of each "channel"
    * D: state dimension, fixed to C * K.
    * L: number of different edge types
    * E: number of edges of a given edge type

    Args:
        node_embeddings: float32 tensor of shape [V, D], the original representation of
            each node in the graph.
        adjacency_lists: List of L adjacency lists, represented as int32 tensors of shape
            [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
            of type l connects node v to node u.
        num_channels: Number of "channels" to split state information into.
        channel_dim: Size of each "channel"
        num_timesteps: Number of repeated applications of this message passing layer.
        use_full_state_for_channel_weights: Flag indicating if the full state is used to
            compute the weights for individual channels, or only the corresponding channel.
        tie_channel_weights: Flag indicating if the weights for computing the per-channel
            linear layer are shared or not.
        activation_function: Type of activation function used.
        message_aggregation_function: Type of aggregation function used for messages.
        normalize_by_num_incoming: Flag indicating if messages should be scaled by 1/(number
            of incoming edges).

    Returns:
        float32 tensor of shape [V, D]
    """
    num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]

    # === Prepare things we need across all timesteps:
    activation_fn = get_activation(activation_function)
    message_aggregation_fn = get_aggregation_function(message_aggregation_function)
    edge_type_to_channel_to_weight_computation_layers = []  # Layers to compute the dynamic computation weights
    edge_type_to_message_targets = []  # List of tensors of message targets

    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        channel_to_weight_computation_layers = []
        for channel in range(num_channels):
            if channel == 0 or not(tie_channel_weights):
                channel_to_weight_computation_layers.append(
                    tf.keras.layers.Dense(
                        units=channel_dim * channel_dim,
                        use_bias=False,
                        kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=1.0 / (channel_dim**2)),
                        activation=activation_fn,
                        name="Edge_%i_Channel_%i_Weight_Computation" % (edge_type_idx, channel)))
            else:  # Case channel > 0 and tie_channel_weights
                channel_to_weight_computation_layers.append(
                    channel_to_weight_computation_layers[-1])
        edge_type_to_channel_to_weight_computation_layers.append(channel_to_weight_computation_layers)

        edge_type_to_message_targets.append(adjacency_list_for_edge_type[:, 1])

    # Let M be the number of messages (sum of all E):
    message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

    cur_node_states = node_embeddings  # Shape [V, D]
    for _ in range(num_timesteps):
        node_states_chunked = tf.reshape(cur_node_states,
                                         shape=(-1, num_channels, channel_dim))  # shape [V, C, K]

        new_node_states_chunked = []  # type: List[tf.Tensor]  # C tensors of shape [V, K]
        for channel_idx in range(num_channels):
            cur_channel_node_states = node_states_chunked[:, channel_idx, :]  # shape [V, K]
            cur_channel_message_per_type = []  # list of tensors of messages of shape [E, K]

            # Collect incoming messages per edge type
            for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
                edge_sources = adjacency_list_for_edge_type[:, 0]
                edge_targets = adjacency_list_for_edge_type[:, 1]
                edge_source_states = \
                    tf.nn.embedding_lookup(params=cur_channel_node_states,
                                           ids=edge_sources)  # Shape [E, K]

                if use_full_state_for_channel_weights:
                    weight_computation_input = cur_node_states
                else:
                    weight_computation_input = cur_channel_node_states
                # TODO: In the tie_channel_weights && use_full_state_for_channel_weights case,
                # this is the same for each channel:
                weight_compute_layer = edge_type_to_channel_to_weight_computation_layers[edge_type_idx][channel_idx]
                edge_weights = weight_compute_layer(weight_computation_input)  # Shape [V, K*K]
                edge_weights = tf.reshape(edge_weights, shape=(-1, channel_dim, channel_dim))  # Shape [V, K, K]
                edge_weights_for_targets = \
                    tf.nn.embedding_lookup(params=edge_weights, ids=edge_targets)  # Shape [E, K, K]

                # Matrix multiply between edge_source_states[v] and edge_weights_for_targets[v]:
                messages = tf.einsum('vi,vij->vj', edge_source_states, edge_weights_for_targets)  # Shape [E, K]
                if normalize_by_num_incoming:
                    num_incoming_to_node_per_message = \
                        tf.nn.embedding_lookup(params=type_to_num_incoming_edges[edge_type_idx, :],
                                               ids=edge_targets)  # Shape [E]
                    messages = tf.expand_dims(1.0 / (num_incoming_to_node_per_message + SMALL_NUMBER), axis=-1) * messages

                cur_channel_message_per_type.append(messages)

            cur_channel_messages = tf.concat(cur_channel_message_per_type, axis=0)  # Shape [M, K]
            cur_channel_aggregated_incoming_messages = \
                message_aggregation_fn(data=cur_channel_messages,
                                       segment_ids=message_targets,
                                       num_segments=num_nodes)  # Shape [V, K]
            cur_channel_aggregated_incoming_messages = activation_fn(cur_channel_aggregated_incoming_messages)

            new_node_states_chunked.append(cur_channel_aggregated_incoming_messages)

        new_node_states = tf.concat(new_node_states_chunked, axis=1)  # Shape [V, C * K]
        cur_node_states = new_node_states

    return cur_node_states
