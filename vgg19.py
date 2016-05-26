import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
import os.path
from utils import max_pool_2x2, conv_layer
from config import VGG_PATH


#average image for the VGG model
average_image = np.array([123.680, 116.779, 103.939], dtype=np.float32)

VGG_layers = (
    ('conv', 'conv1_1', 64),  ('relu', 'relu1_1'), ('conv', 'conv1_2', 64),  ('relu', 'relu1_2'), ('pool', 'pool1'),

    ('conv', 'conv2_1', 128), ('relu', 'relu2_1'), ('conv', 'conv2_2', 128), ('relu', 'relu2_2'), ('pool', 'pool2'),

    ('conv', 'conv3_1', 256), ('relu', 'relu3_1'), ('conv', 'conv3_2', 256), ('relu', 'relu3_2'),
    ('conv', 'conv3_3', 256), ('relu', 'relu3_3'), ('conv', 'conv3_4', 256), ('relu', 'relu3_4'), ('pool', 'pool3'),

    ('conv', 'conv4_1', 512), ('relu', 'relu4_1'), ('conv', 'conv4_2', 512), ('relu', 'relu4_2'),
    ('conv', 'conv4_3', 512), ('relu', 'relu4_3'), ('conv', 'conv4_4', 512), ('relu', 'relu4_4'), ('pool', 'pool4'),

    ('conv', 'conv5_1', 512), ('relu', 'relu5_1'), ('conv', 'conv5_2', 512), ('relu', 'relu5_2'),
    ('conv', 'conv5_3', 512), ('relu', 'relu5_3'), ('conv', 'conv5_4', 512), ('relu', 'relu5_4'), ('pool', 'pool5'),

    ('fc', 'fc6', 4096, 7), ('relu', 'relu6'), ('dropout', 'dropout1'),
    ('fc', 'fc7', 4096, 1), ('relu', 'relu7'), ('dropout', 'dropout2'),
    )


def get_VGG_layers(images_ph, dropout_keep_prob_ph, train_fc_layers=False):
    """
    Args:
        images: Images placeholder in NHWC.
    Returns:
        dict of tensors, keys are original layer names.
    """
    x = images_ph
    layers = dict()
    for layer in VGG_layers:
        layer_type = layer[0]
        layer_name = layer[1]
        if layer_type == 'conv':
            with tf.variable_scope(layer_name) as scope:
                x = conv_layer(x, layer[2], 3, layer_name, train=False)
        elif layer_type == 'pool':
            with tf.variable_scope(layer_name) as scope:
                x = max_pool_2x2(x, layer_name)
        elif layer_type == 'fc':
            with tf.variable_scope(layer_name) as scope:
                x = conv_layer(x, layer[2], layer[3], layer_name, train=train_fc_layers)
        elif layer_type == 'relu':
            x = tf.nn.relu(x, layer_name)
        elif layer_type == 'dropout':
            x = tf.nn.dropout(x, dropout_keep_prob_ph, name=layer_name)

        layers[layer_name] = x

    return layers


def get_initialize_op_for_VGG():
    data = scipy.io.loadmat(VGG_PATH)
    init_ops = []
    for layer in data['layers'][0,:]:
        layer_type = layer['type'][0,0][0]
        if layer_type == 'conv':
            layer_name = layer['name'][0,0][0]
            if layer_name != 'fc8':
                weights, biases = layer['weights'][0,0][0]
                biases = biases[:,0]
                weights = np.transpose(weights, (1, 0, 2, 3))
                with tf.variable_scope(layer_name, reuse=True):
                    w = tf.get_variable('weights')
                    b = tf.get_variable('biases')
                    init_ops.append(w.assign(tf.constant(weights)))
                    init_ops.append(b.assign(tf.constant(biases)))
    return init_ops
