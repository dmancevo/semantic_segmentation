import tensorflow as tf

def _weight_variable(name, shape, train=True):
    return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1), trainable=train)


def _bias_variable(name, shape, train=True):
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape), trainable=train)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(x, out_channels, filter_side, name, train=True, padding='SAME'):
    in_channels = x.get_shape().as_list()[3]
    kernel = _weight_variable('weights', shape=[filter_side, filter_side, in_channels, out_channels], train=train)
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding=padding)
    biases = _bias_variable('biases', shape=[out_channels], train=train)
    return tf.nn.bias_add(conv, biases)