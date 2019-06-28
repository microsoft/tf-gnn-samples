from typing import List, Optional
import tensorflow as tf


from utils import get_activation, get_aggregation_function, SMALL_NUMBER


def sparse_gnn_film_layer(node_embeddings: tf.Tensor,
                          adjacency_lists: List[tf.Tensor],
                          type_to_num_incoming_edges: tf.Tensor,
                          state_dim: Optional[int],
                          num_timesteps: int = 1,
                          activation_function: Optional[str] = "ReLU",
                          message_aggregation_function: str = "sum",
                          normalize_by_num_incoming: bool = False,
                          ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing modulated by the target state.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sum_\ell
                     \sum_{(u, v) \in A_\ell}
                        \sigma(1/c_{v,\ell} * \alpha_{\ell,v} * (W_\ell * h^t_u) + \beta_{\ell,v})
        \alpha_{\ell,v} := F_{\ell,\alpha} * h^t_v
        \beta_{\ell,v} := F_{\ell,\beta} * h^t_v
        c_{\v,\ell} is usually 1 (but could also be the number of incoming edges).
    The learnable parameters of this are the W_\ell, F_{\ell,\alpha}, F_{\ell,\beta} \in R^{D, D}.

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
    edge_type_to_film_computation_layers = []  # Layers to compute the \beta/\gamma weights for FiLM
    edge_type_to_message_targets = []  # List of tensors of message targets
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        edge_type_to_message_transformation_layers.append(
            tf.keras.layers.Dense(units=state_dim,
                                  use_bias=False,
                                  activation=None,  # Activation only after FiLM modulation
                                  name="Edge_%i_Weight" % edge_type_idx))
        edge_type_to_film_computation_layers.append(
            tf.keras.layers.Dense(units=2 * state_dim,  # Computes \gamma, \beta in one go
                                  use_bias=False,
                                  activation=None,
                                  name="Edge_%i_FiLM_Computations" % edge_type_idx))
        edge_type_to_message_targets.append(adjacency_list_for_edge_type[:, 1])

    # Let M be the number of messages (sum of all E):
    message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

    cur_node_states = node_embeddings
    for _ in range(num_timesteps):
        messages_per_type = []  # list of tensors of messages of shape [E, D]
        # Collect incoming messages per edge type
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]
            edge_source_states = \
                tf.nn.embedding_lookup(params=cur_node_states,
                                       ids=edge_sources)  # Shape [E, D]
            messages = edge_type_to_message_transformation_layers[edge_type_idx](edge_source_states)  # Shape [E, D]

            if normalize_by_num_incoming:
                per_message_num_incoming_edges = \
                    tf.nn.embedding_lookup(params=type_to_num_incoming_edges[edge_type_idx, :],
                                           ids=edge_targets)  # Shape [E, H]
                messages = tf.expand_dims(1.0 / (per_message_num_incoming_edges + SMALL_NUMBER), axis=-1) * messages

            film_weights = edge_type_to_film_computation_layers[edge_type_idx](cur_node_states)
            per_message_film_weights = \
                tf.nn.embedding_lookup(params=film_weights, ids=edge_targets)
            per_message_film_gamma_weights = per_message_film_weights[:, :state_dim]  # Shape [E, D]
            per_message_film_beta_weights = per_message_film_weights[:, state_dim:]  # Shape [E, D]

            modulated_messages = per_message_film_gamma_weights * messages + per_message_film_beta_weights
            messages_per_type.append(modulated_messages)

        all_messages = tf.concat(messages_per_type, axis=0)  # Shape [M, D]
        all_messages = activation_fn(all_messages)  # Shape [M, D]
        aggregated_messages = \
            message_aggregation_fn(data=all_messages,
                                   segment_ids=message_targets,
                                   num_segments=num_nodes)  # Shape [V, D]
        new_node_states = aggregated_messages
        # new_node_states = activation_fn(new_node_states)

        cur_node_states = tf.contrib.layers.layer_norm(new_node_states)

    return cur_node_states
