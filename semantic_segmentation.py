import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf

def center(image, mean_pixel):
  return image - mean_pixel

def un_center(image, mean_pixel):
  return image + mean_pixel

def _weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def _bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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

#Center image and add batch dimension
im = center(im,mean_pixel)
im = im[np.newaxis,:,:,:]

#Image representation on layer relu1_1
print net['relu1_1'].eval(feed_dict={input_image: im},session=sess).shape

  
##########################################
#########ADD DECONVOLUTION LAYERS#########
##########################################

#input = (1, 523, 769, 64) from layer relu1_1 (later we need to check the input from images on our data set)
#output = (input + stride - 1)
deconv = tf.nn.conv2d_transpose(value=net['relu1_1'], filter=_weight_variable((16, 16, 3, 64)),output_shape=(1, 523, 769, 3),
strides=(1, 1, 1, 1), padding='SAME')

sess.run(tf.initialize_all_variables())

out = deconv.eval(feed_dict={input_image: im},session=sess)[0]
scipy.misc.imsave('deconv_out.png',out)



##########################################
########TRAIN DECONVOLUTION LAYERS########
##########################################