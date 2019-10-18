from typing import Optional, Callable, Union, List

import tensorflow as tf


BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7


def get_gated_unit(units: int, gated_unit: str, activation_function: str):
    activation_fn = get_activation(activation_function)
    gated_unit_name = gated_unit.lower()
    if gated_unit_name == 'rnn':
        return tf.keras.layers.SimpleRNNCell(units, activation=activation_fn)
    if gated_unit_name == 'gru':
        return tf.keras.layers.GRUCell(units, activation=activation_fn)
    if gated_unit_name == 'lstm':
        return tf.keras.layers.LSTMCell(units, activation=activation_fn)
    else:
        raise Exception("Unknown RNN cell type '%s'." % gated_unit)


def get_aggregation_function(aggregation_fun: Optional[str]):
    if aggregation_fun in ['sum', 'unsorted_segment_sum']:
        return tf.unsorted_segment_sum
    if aggregation_fun in ['max', 'unsorted_segment_max']:
        return tf.unsorted_segment_max
    if aggregation_fun in ['mean', 'unsorted_segment_mean']:
        return tf.unsorted_segment_mean
    if aggregation_fun in ['sqrt_n', 'unsorted_segment_sqrt_n']:
        return tf.unsorted_segment_sqrt_n
    else:
        raise ValueError("Unknown aggregation function '%s'!" % aggregation_fun)


def get_activation(activation_fun: Optional[str]):
    if activation_fun is None:
        return None
    activation_fun = activation_fun.lower()
    if activation_fun == 'linear':
        return None
    if activation_fun == 'tanh':
        return tf.tanh
    if activation_fun == 'relu':
        return tf.nn.relu
    if activation_fun == 'leaky_relu':
        return tf.nn.leaky_relu
    if activation_fun == 'elu':
        return tf.nn.elu
    if activation_fun == 'selu':
        return tf.nn.selu
    if activation_fun == 'gelu':
        def gelu(input_tensor):
            cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
            return input_tensor * cdf
        return gelu
    else:
        raise ValueError("Unknown activation function '%s'!" % activation_fun)


def micro_f1(logits, labels):
    # Everything on int, because who trusts float anyway?
    predicted = tf.round(tf.nn.sigmoid(logits))
    predicted = tf.cast(predicted, dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)

    true_pos = tf.count_nonzero(predicted * labels)
    false_pos = tf.count_nonzero(predicted * (labels - 1))
    false_neg = tf.count_nonzero((predicted - 1) * labels)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return tf.cast(fmeasure, tf.float32)


class MLP(object):
    def __init__(self,
                 out_size: int,
                 hidden_layers: Union[List[int], int] = 1,
                 use_biases: bool = False,
                 activation_fun: Optional[Callable[[tf.Tensor], tf.Tensor]] = tf.nn.relu,
                 dropout_rate: Union[float, tf.Tensor] = 0.0,
                 name: Optional[str] = "MLP",
                 ):
        """
        Create new MLP with given number of hidden layers.

        Arguments:
            out_size: Dimensionality of output.
            hidden_layers: Either an integer determining number of hidden layers, who will have out_size units each;
                or list of integers whose lengths determines the number of hidden layers and whose contents the
                number of units in each layer.
            use_biases: Flag indicating use of bias in fully connected layers.
            activation_fun: Activation function applied between hidden layers (NB: the output of the MLP
                is always the direct result of a linear transformation)
            dropout_rate: Dropout applied to inputs of each MLP layer.
        """
        if isinstance(hidden_layers, int):
            hidden_layer_sizes = [out_size] * hidden_layers
        else:
            hidden_layer_sizes = hidden_layers

        if len(hidden_layer_sizes) > 1:
            assert activation_fun is not None, "Multiple linear layers without an activation"

        self.__dropout_rate = dropout_rate
        self.__name = name
        with tf.variable_scope(self.__name):
            self.__layers = []  # type: List[tf.layers.Dense]
            for hidden_layer_size in hidden_layer_sizes:
                self.__layers.append(tf.layers.Dense(units=hidden_layer_size,
                                                     use_bias=use_biases,
                                                     activation=activation_fun))
            # Output layer:
            self.__layers.append(tf.layers.Dense(units=out_size,
                                                 use_bias=use_biases,
                                                 activation=None))

    def __call__(self, input: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope(self.__name):
            activations = input
            for layer in self.__layers[:-1]:
                activations = tf.nn.dropout(activations, rate=self.__dropout_rate)
                activations = layer(activations)
            return self.__layers[-1](activations)
