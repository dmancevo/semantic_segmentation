import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf
from load_images import train_test

def _weight_variable(shape):
  '''weight variable'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def _bias_variable(shape):
  '''bias variable'''
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def _conv_layer(input, weights, bias):
  '''convolution layer'''
  conv = tf.nn.conv2d(input, tf.constant(weights),
    strides=(1, 1, 1, 1), padding='SAME')
  return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
  '''pool layer'''
  return tf.nn.max_pool(input, ksize=(1, 2, 2, 1),
    strides=(1, 2, 2, 1), padding='SAME')
          
def _deconv_layer(input_layer, filter, input_image, strides):
  '''deconvolution layer'''
  
  deconv = tf.nn.conv2d_transpose(value=input_layer, filter=_weight_variable(filter),
    output_shape=tf.pack((1,tf.shape(input_image)[1],tf.shape(input_image)[2],1)),
    strides=strides, padding='SAME')
    
  bias = _bias_variable((filter[2],)) #bias shape should match the output channels of the layer.
  return tf.nn.bias_add(deconv, bias)


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
      'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
  )
  

#Load vgg network weights from .mat file
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
data = scipy.io.loadmat(VGG_PATH)
weights = data['layers'][0]

#Load mean pixel values so we can center pictures
#(This is done on the VGG paper as well)
mean = data['normalization'][0][0][0]
mean_pixel = np.mean(mean, axis=(0, 1))

#Keep track of all the network layers so we can extract output at arbitrary locations.
net = {}

#Set up the graph and the session
g = tf.Graph()
g.as_default()
g.device('/gpu:0') #Device can be either cpu or gpu
sess=tf.Session()

#Placeholder for the image
input_image = tf.placeholder(tf.float32, shape=(None, None, None, 3))

#Center image
centered_image = tf.sub(input_image, tf.constant(mean_pixel, dtype=tf.float32))
current = centered_image

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

  
##########################################
#########ADD DECONVOLUTION LAYERS#########
##########################################


#Deconvolutions for up-sampling
deconv3 = tf.nn.relu(_deconv_layer(input_layer=net['pool3'], filter=(8, 8, 1, 256),
  input_image=input_image, strides=(1, 8, 8, 1)))

deconv4 = tf.nn.relu(_deconv_layer(input_layer=net['pool4'], filter=(16, 16, 1, 512),
  input_image=input_image, strides=(1, 16, 16, 1)))

deconv5 = tf.nn.relu(_deconv_layer(input_layer=net['pool5'], filter=(32, 32, 1, 512),
  input_image=input_image, strides=(1, 32, 32, 1)))
  
#Concatenate them, one deconvolution per channel
deconvs = tf.concat(3,(deconv3, deconv4, deconv5))

#One last convolution to rule them all
conv    = tf.nn.bias_add(tf.nn.conv2d(deconvs, _weight_variable((1,1,3,3)),
  strides=(1,1,1,1), padding="SAME"), _bias_variable((3,)))

#Process one image
im, se = train_test('2010_003468')
im = im[np.newaxis,:,:,:]

sess.run(tf.initialize_all_variables())

out = conv.eval(feed_dict={input_image: im},session=sess)
print out.shape
scipy.misc.imsave("out.png",out[0])


##########################################
########TRAIN DECONVOLUTION LAYERS########
##########################################



sess.close()