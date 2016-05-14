import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf

def _conv_layer(input, weights, bias):
  conv = tf.nn.conv2d(input, tf.constant(weights),
    strides=(1, 1, 1, 1), padding='SAME')
  return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
  return tf.nn.max_pool(input, ksize=(1, 2, 2, 1),
    strides=(1, 2, 2, 1), padding='SAME')
          
#Still need to set this one up
def _deconv_layer(value, filter, output_shape):
  deconv = tf.nn.conv2d_transpose(value, filter, output_shape,
    strides, padding='SAME')


##########################################
#########LOAD PRE-TRAINED NETWORK#########
##########################################

#Network architecture
layers = (
      'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
      'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
      'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
      'relu5_3', 'conv5_4', 'relu5_4'
  )
  

#Load vgg network weights from .mat file
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
data = scipy.io.loadmat(VGG_PATH)
weights = data['layers'][0]

#Create placeholder for image (I'm using an arbitrary image here)
im = scipy.misc.imread("bici.jpg").astype(np.float)
shape = (1,) + im.shape

#Keep track of all the network layers so we can extract output at arbitrary locations.
net = {}

#Set up the graph and the session
g = tf.Graph()
g.as_default()
g.device('/gpu:0') #Device can be either cpu or gpu
sess=tf.Session()

#Placeholder for the image
input_image = tf.placeholder('float', shape=shape)
current = input_image

for i, name in enumerate(layers):
  kind = name[:4]
  if kind == 'conv':
    kernels, bias = weights[i][0][0][0][0]
    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]
    kernels = np.transpose(kernels, (1, 0, 2, 3))
    bias = bias.reshape(-1)
    current = _conv_layer(current, kernels, bias)
  elif kind == 'relu':
    current = tf.nn.relu(current)
  elif kind == 'pool':
    current = _pool_layer(current)
  net[name] = current

#Add one more dimention to match tensor input dimensions
im = im[np.newaxis,:,:,:]

#Store image representation on layer
print net['relu5_4'].eval(feed_dict={input_image: im},session=sess).shape
  
  
##########################################
#########ADD DECONVOLUTION LAYERS#########
##########################################

deConv_net = {}





##########################################
########TRAIN DECONVOLUTION LAYERS########
##########################################