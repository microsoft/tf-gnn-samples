import tensorflow as tf


class DenseWithWeightDropout(tf.keras.layers.Dense):
    def __init__(self, dropout_ratio: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_ratio = dropout_ratio

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = tf.nn.dropout(self.kernel, rate=self.dropout_ratio)
        if self.use_bias:
            self.bias = tf.nn.dropout(self.bias, rate=self.dropout_ratio)

    def get_config(self):
        config = super().get_config()
        config['dropout_ratio'] = self.dropout_ratio
        return config
