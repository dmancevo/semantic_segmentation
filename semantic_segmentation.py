import numpy as np
import scipy.misc
import tensorflow as tf
import vgg

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

im = scipy.misc.imread("bici.jpg").astype(np.float)

shape = (1,) + im.shape
image_features = {}

g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess: #Device can be either cpu or gpu
    
  #Create placeholder for image
  image = tf.placeholder('float', shape=shape)
  
  #Read network from .mat file and return along with image mean pixel
  net, mean_pixel = vgg.net(VGG_PATH, image)
  
  #Pre-process image by normalizing it (zero mean)
  im_pre = np.array([vgg.preprocess(im, mean_pixel)])
  
  #Store image representation on layer
  image_features['relu4_2'] = net['relu4_2'].eval(feed_dict={image: im_pre})
  

representation = image_features['relu4_2']
print representation.shape