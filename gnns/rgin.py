from typing import List, Optional
import tensorflow as tf

from utils import get_activation, get_aggregation_function, MLP


def sparse_rgin_layer(
        node_embeddings: tf.Tensor,
        adjacency_lists: List[tf.Tensor],
        state_dim: Optional[int],
        num_timesteps: int = 1,
        activation_function: Optional[str] = "ReLU",
        message_aggregation_function: str = "sum",
        use_target_state_as_input: bool = False,
        num_MLP_hidden_layers: int = 1,
        learn_epsilon: bool = True,
        ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing using MLPs for state updates
    and message computation.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := MLP_{out}((1 + \epsilon) * MLP_{self}(h^t_v)
                               + \sum_\ell \sum_{(u, v) \in A_\ell} MLP_\ell(h^t_u))
    The learnable parameters of this are the MLPs and (if enabled) epsilon.
    This is derived from Cor. 6 of arXiv:1810.00826, instantiating the functions f, \phi
    with _separate_ MLPs. This is more powerful than the GIN formulation in Eq. (4.1) of
    arXiv:1810.00826, as we want to be able to distinguish graphs of the form
     G_1 = (V={1, 2, 3}, E_1={(1, 2)}, E_2={(3, 2)})
    and
     G_2 = (V={1, 2, 3}, E_1={(3, 2)}, E_2={(1, 2)})
    from each other. If we would treat all edges the same,
    G_1.E_1 \cup G_1.E_2 == G_2.E_1 \cup G_2.E_2 would imply that the two graphs
    become indistuingishable.
    Hence, we introduce per-edge-type MLPs, which also means that we have to drop
    the optimisation of modelling f \circ \phi by a single MLP used in the original
    GIN formulation.

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
        activation_function: Type of activation function used.
        message_aggregation_function: Type of aggregation function used for messages.
        use_target_state_as_input: Flag indicating if the edge MLP should consume both
            source and target state (True) or only source state (False).
        num_MLP_hidden_layers: Number of hidden layers of the MLPs.
        learn_epsilon: Flag indicating if the value of epsilon should be learned. If
            False, epsilon defaults to 0.

    Returns:
        float32 tensor of shape [V, state_dim]
    """
    num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]
    if state_dim is None:
        state_dim = tf.shape(node_embeddings, out_type=tf.int32)[1]

    # === Prepare things we need across all timesteps:
    activation_fn = get_activation(activation_function)
    message_aggregation_fn = get_aggregation_function(message_aggregation_function)
    aggregation_MLP = MLP(out_size=state_dim,
                          hidden_layers=num_MLP_hidden_layers,
                          activation_fun=activation_fn,
                          name="Aggregation_MLP")
    edge_type_to_edge_mlp = []  # MLPs to compute the edge messages
    edge_type_to_message_targets = []  # List of tensors of message targets
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        edge_type_to_edge_mlp.append(
            MLP(out_size=state_dim,
                hidden_layers=num_MLP_hidden_layers,
                activation_fun=activation_fn,
                name="Edge_%i_MLP" % edge_type_idx))
        edge_type_to_message_targets.append(adjacency_list_for_edge_type[:, 1])
    # Initialize epsilon: Note that we merge the 1 + \epsilon from the Def. above:
    if learn_epsilon:
        epsilon_plus_one = tf.get_variable("epsilon", shape=(), dtype=tf.float32, initializer=tf.ones_initializer, trainable=True)
    else:
        epsilon_plus_one = 1
    self_loop_MLP = MLP(out_size=state_dim,
                        hidden_layers=num_MLP_hidden_layers,
                        activation_fun=activation_fn,
                        name="SelfLoop_MLP")

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

            edge_mlp_inputs = edge_source_states
            if use_target_state_as_input:
                edge_target_states = \
                    tf.nn.embedding_lookup(params=cur_node_states,
                                           ids=edge_targets)  # Shape [E, D]
                edge_mlp_inputs = tf.concat([edge_source_states, edge_target_states],
                                            axis=1)  # Shape [E, 2*D]

            messages = edge_type_to_edge_mlp[edge_type_idx](edge_mlp_inputs)  # Shape [E, D]
            messages_per_type.append(messages)

        all_messages = tf.concat(messages_per_type, axis=0)  # Shape [M, D]
        all_messages = activation_fn(all_messages)  # Shape [M, D]  (Apply nonlinearity to Edge-MLP outputs as well)
        aggregated_messages = \
            message_aggregation_fn(data=all_messages,
                                   segment_ids=message_targets,
                                   num_segments=num_nodes)  # Shape [V, D]

        new_node_states = aggregated_messages
        new_node_states += epsilon_plus_one * activation_fn(self_loop_MLP(cur_node_states))
        new_node_states = aggregation_MLP(new_node_states)
        new_node_states = activation_fn(new_node_states)  # Note that the final MLP layer has no activation, so we do that here explicitly
        new_node_states = tf.contrib.layers.layer_norm(new_node_states)
        cur_node_states = new_node_states

    return cur_node_states
