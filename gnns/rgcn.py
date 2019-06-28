from typing import List, Optional

import tensorflow as tf

from utils import get_activation, get_aggregation_function, SMALL_NUMBER


def sparse_rgcn_layer(node_embeddings: tf.Tensor,
                      adjacency_lists: List[tf.Tensor],
                      type_to_num_incoming_edges: tf.Tensor,
                      state_dim: Optional[int],
                      num_timesteps: int = 1,
                      activation_function: Optional[str] = "tanh",
                      message_aggregation_function: str = "sum",
                      normalize_by_num_incoming: bool = True,
                      use_both_source_and_target: bool = False,
                      ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing.
    This implements the R-GCN model (Schlichtkrull et al., https://arxiv.org/pdf/1703.06103.pdf)
    for the case of few relations / edge types, i.e., we do not use the dimensionality-reduction
    tricks from section 2.2 of that paper.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sigma(\sum_\ell
                            \sum_{(u, v) \in A_\ell}
                               1/c_{v,\ell} * (W_\ell * h^t_u))
    c_{\v,\ell} is usually the number of \ell edges going into v.
    The learnable parameters of this are the W_\ell \in R^{D,D}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * D: state dimension
    * L: number of different edge types
    * E: number of edges of a given edge type

    Arguments:
        node_embeddings: float32 tensor of shape [V, D], the original representation of
            each node in the graph.
        adjacency_lists: List of L adjacency lists, represented as int32 tensors of shape
            [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
            of type l connects node v to node u.
        type_to_num_incoming_edges: float32 tensor of shape [L, V] representing the number
            of incoming edges of a given type. Concretely, type_to_num_incoming_edges[l, v]
            is the number of edge of type l connecting to node v.
        state_dim: Optional size of output dimension of the GNN layer. If not set, defaults
            to D, the dimensionality of the input. If different from the input dimension,
            parameter num_timesteps has to be 1.
        num_timesteps: Number of repeated applications of this message passing layer.
        activation_function: Type of activation function used.
        message_aggregation_function: Type of aggregation function used for messages.
        normalize_by_num_incoming: Flag indicating if messages should be scaled by 1/(number
            of incoming edges).

    Returns:
        float32 tensor of shape [V, state_dim]
    """
    num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]
    if state_dim is None:
        state_dim = tf.shape(node_embeddings, out_type=tf.int32)[1]

    # === Prepare things we need across all timesteps:
    activation_fn = get_activation(activation_function)
    message_aggregation_fn = get_aggregation_function(message_aggregation_function)
    edge_type_to_message_transformation_layers = []  # Layers to compute the message from a source state
    edge_type_to_message_targets = []  # List of tensors of message targets
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        edge_type_to_message_transformation_layers.append(
            tf.keras.layers.Dense(units=state_dim,
                                  use_bias=False,
                                  activation=None,
                                  name="Edge_%i_Weight" % edge_type_idx))
        edge_type_to_message_targets.append(adjacency_list_for_edge_type[:, 1])

    # Let M be the number of messages (sum of all E):
    message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

    cur_node_states = node_embeddings
    for _ in range(num_timesteps):
        messages_per_type = []  # list of tensors of messages of shape [E, H]
        # Collect incoming messages per edge type
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]
            edge_source_states = \
                tf.nn.embedding_lookup(params=cur_node_states,
                                       ids=edge_sources)  # Shape [E, H]

            if use_both_source_and_target:
                edge_target_states = \
                    tf.nn.embedding_lookup(params=cur_node_states,
                                           ids=edge_targets)  # Shape [E, H]
                edge_state_pairs = tf.concat([edge_source_states, edge_target_states], axis=-1)  # Shape [E, 2H]
                messages = edge_type_to_message_transformation_layers[edge_type_idx](edge_state_pairs)  # Shape [E, H]
            else:
                messages = edge_type_to_message_transformation_layers[edge_type_idx](edge_source_states)  # Shape [E, H]

            if normalize_by_num_incoming:
                num_incoming_to_node_per_message = \
                    tf.nn.embedding_lookup(params=type_to_num_incoming_edges[edge_type_idx, :],
                                           ids=edge_targets)  # Shape [E, H]
                messages = tf.expand_dims(1.0 / (num_incoming_to_node_per_message + SMALL_NUMBER), axis=-1) * messages

            messages_per_type.append(messages)

        cur_messages = tf.concat(messages_per_type, axis=0)  # Shape [M, H]
        aggregated_messages = \
            message_aggregation_fn(data=cur_messages,
                                   segment_ids=message_targets,
                                   num_segments=num_nodes)  # Shape [V, H]

        new_node_states = activation_fn(aggregated_messages)  # Shape [V, H]
        cur_node_states = new_node_states

    return cur_node_states
