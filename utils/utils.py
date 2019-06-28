from typing import Optional

import numpy as np
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
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, rate=1.0 - self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden
