from typing import List, Optional

import tensorflow as tf

from utils import get_gated_unit, get_aggregation_function


def sparse_ggnn_layer(node_embeddings: tf.Tensor,
                      adjacency_lists: List[tf.Tensor],
                      state_dim: Optional[int],
                      num_timesteps: int = 1,
                      gated_unit_type: str = "gru",
                      activation_function: str = "tanh",
                      message_aggregation_function: str = "sum"
                      ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing and gated units on the nodes.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := Cell(h^t_v, \sum_\ell
                                 \sum_{(u, v) \in A_\ell}
                                     W_\ell * h^t_u)
    The learnable parameters of this are the recurrent Cell and the W_\ell \in R^{D,D}.

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
        state_dim: Optional size of output dimension of the GNN layer. If not set, defaults
            to D, the dimensionality of the input. If different from the input dimension,
            parameter num_timesteps has to be 1.
        num_timesteps: Number of repeated applications of this message passing layer.
        gated_unit_type: Type of the recurrent unit used (one of RNN, GRU and LSTM).
        activation_function: Type of activation function used.
        message_aggregation_function: Type of aggregation function used for messages.

    Returns:
        float32 tensor of shape [V, state_dim]
    """
    num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]
    if state_dim is None:
        state_dim = tf.shape(node_embeddings, out_type=tf.int32)[1]

    # === Prepare things we need across all timesteps:
    message_aggregation_fn = get_aggregation_function(message_aggregation_function)
    gated_cell = get_gated_unit(state_dim, gated_unit_type, activation_function)
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
        messages = []  # list of tensors of messages of shape [E, D]
        message_source_states = []  # list of tensors of edge source states of shape [E, D]

        # Collect incoming messages per edge type
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_source_states = tf.nn.embedding_lookup(params=cur_node_states,
                                                        ids=edge_sources)  # Shape [E, D]
            all_messages_for_edge_type = \
                edge_type_to_message_transformation_layers[edge_type_idx](edge_source_states)  # Shape [E,D]
            messages.append(all_messages_for_edge_type)
            message_source_states.append(edge_source_states)

        messages = tf.concat(messages, axis=0)  # Shape [M, D]
        aggregated_messages = \
            message_aggregation_fn(data=messages,
                                   segment_ids=message_targets,
                                   num_segments=num_nodes)  # Shape [V, D]

        # pass updated vertex features into RNN cell
        new_node_states = gated_cell(aggregated_messages, [cur_node_states])[0]  # Shape [V, D]
        cur_node_states = new_node_states

    return cur_node_states
