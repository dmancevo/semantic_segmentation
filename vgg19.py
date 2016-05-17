import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
import os.path

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
BATCH_SIZE = 1
CHECKPOINT_DIR = "/Users/Olga/Education/KTH/DD2427_IBRC/Project/semantic_segmentation/ckpt_vgg"
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

layers = (
    ('conv', 'conv1_1', 64),  ('relu', 'relu1_1'), ('conv', 'conv1_2', 64),  ('relu', 'relu1_2'), ('pool', 'pool1'),

    ('conv', 'conv2_1', 128), ('relu', 'relu2_1'), ('conv', 'conv2_2', 128), ('relu', 'relu2_2'), ('pool', 'pool2'),

    ('conv', 'conv3_1', 256), ('relu', 'relu3_1'), ('conv', 'conv3_2', 256), ('relu', 'relu3_2'),
    ('conv', 'conv3_3', 256), ('relu', 'relu3_3'), ('conv', 'conv3_4', 256), ('relu', 'relu3_4'), ('pool', 'pool3'),

    ('conv', 'conv4_1', 512), ('relu', 'relu4_1'), ('conv', 'conv4_2', 512), ('relu', 'relu4_2'),
    ('conv', 'conv4_3', 512), ('relu', 'relu4_3'), ('conv', 'conv4_4', 512), ('relu', 'relu4_4'), ('pool', 'pool4'),

    ('conv', 'conv5_1', 512), ('relu', 'relu5_1'), ('conv', 'conv5_2', 512), ('relu', 'relu5_2'),
    ('conv', 'conv5_3', 512), ('relu', 'relu5_3'), ('conv', 'conv5_4', 512), ('relu', 'relu5_4'), ('pool', 'pool5'),

    ('fc', 'fc6', 4096), ('relu', 'relu6'),
    ('fc', 'fc7', 4096), ('relu', 'relu7'),
    ('fc', 'fc8', 1000),
    ('softmax', 'prob')
    )


def _weight_variable(name, shape):
    #return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    return tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=0.1))
    


def _bias_variable(name, shape):
    #return tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))


def _max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def _conv_3x3(x, out_channels, name):
    in_channels = x.get_shape().as_list()[3]
    kernel = _weight_variable('weights', shape=[3, 3, in_channels, out_channels])
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', shape=[out_channels])
    return tf.nn.bias_add(conv, biases)

def _fc(x, out_channels, name):
    shape = x.get_shape().as_list()
    in_channels = shape[3]
    kernel = _weight_variable('weights', shape=[shape[1], shape[2], in_channels, out_channels])
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _bias_variable('biases', shape=[out_channels])
    return tf.nn.bias_add(conv, biases)


def get_VGG_inference(images):
    """
    Args:
        images: Images placeholder in NHWC.
    Returns:
        Output tensor.
    """
    x = images
    for layer in layers:
        layer_type = layer[0]
        layer_name = layer[1]
        if layer_type == 'conv':
            with tf.variable_scope(layer_name) as scope:
                out_channels = layer[2]
                x = _conv_3x3(x, out_channels, layer_name)
        elif layer_type == 'pool':
            x = _max_pool_2x2(x, layer_name)
        elif layer_type == 'fc':
            with tf.variable_scope(layer_name) as scope:
                out_units = layer[2]
                x = _fc(x, out_units, layer_name)
        elif layer_type == 'relu':
            x = tf.nn.relu(x, layer_name)
        elif layer_type == 'softmax':
            x = tf.reshape(x, [BATCH_SIZE, -1])
            x = tf.nn.softmax(x, layer_name)

    return x

def get_initialize_op_for_VGG():
    
    data = scipy.io.loadmat(VGG_PATH)
    average_image = data['meta']['normalization'][0,0]['averageImage'][0,0]
    class_descriptions = data['meta']['classes'][0,0]['description'][0,0]
    init_ops = []
    for layer in data['layers'][0,:]:
        layer_type = layer['type'][0,0][0]
        if layer_type == 'conv':
            layer_name = layer['name'][0,0][0]
            weights, biases = layer['weights'][0,0][0]
            biases = biases[:,0]
            #print(layer_type, layer_name, weights.shape, biases.shape)
            with tf.variable_scope(layer_name, reuse=True):
                w = tf.get_variable('weights')
                b = tf.get_variable('biases')

                init_ops.append(w.assign(tf.constant(weights)))
                init_ops.append(b.assign(tf.constant(biases)))
    return init_ops


if __name__ == '__main__':
    images_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    y_sotfmax = get_VGG_inference(images_ph)
    
    im = scipy.misc.imread("flower.jpg").astype(np.float)
    image = np.reshape(im, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("loaded")
        elif os.path.isfile(VGG_PATH):
            init_ops = get_initialize_op_for_VGG()
            sess.run(init_ops)
            print("initialized")
            saver.save(sess, CHECKPOINT_DIR + "/vgg.ckpt")
            print("saved")
        else:
            sess.run(tf.initialize_all_variables())
            print("initialized with defaults")

        res = sess.run(y_sotfmax, feed_dict={images_ph: image})
        print np.sum(res)
        with tf.variable_scope('conv1_1', reuse=True):
            #tf.get_variable('weights', initializer=tf.constant(weights))
            bs = tf.get_variable('biases').eval()
            print(bs)
    
    #print(map(lambda x: x.name, tf.all_variables()))
    #print(map(lambda x: x.name, tf.get_default_graph().get_operations()))
