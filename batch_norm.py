#http://stackoverflow.com/questions/36978798/batch-normalization-in-tensorflow
#https://github.com/tensorflow/models/blob/master/inception/inception/slim/ops.py#L116

#The code below was modified from the two sources above (credit to the original authors).

import tensorflow as tf
from tensorflow.python.training import moving_averages


def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               is_training=True,
               trainable=True,
               restore=True,
               scope=None,
               reuse=None):
  """Adds a Batch Normalization layer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    center: If True, subtract beta. If False, beta is not created and ignored.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    activation: activation function.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_op_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.
  """
  inputs_shape = inputs.get_shape()
  with tf.variable_op_scope([inputs], scope, 'BatchNorm', reuse=reuse):
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    
    # Allocate parameters for the beta and gamma of the normalization.
    beta = tf.Variable(tf.zeros(params_shape), name='beta')
    gamma = beta = tf.Variable(tf.ones(params_shape), name='gamma')
    
    moving_mean = tf.Variable(tf.zeros(params_shape), name='moving_mean',
                                   trainable=False)
    moving_variance = tf.Variable(tf.ones(params_shape),
                                       name='moving_variance',
                                       trainable=False)
                                     
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)
      
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)

    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance
      
    # Normalize the activations.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())

    return outputs