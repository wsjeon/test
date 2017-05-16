# See link(http://blog.evjang.com/2016/11/tutorial-categorical-variational.html).


import tensorflow as tf
import numpy as np

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

def _fc_variable(weight_shape):
  input_channels  = weight_shape[0]
  output_channels = weight_shape[1]
  d = 1.0 / np.sqrt(input_channels)
  bias_shape = [output_channels]
  weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
  bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
  return weight, bias

# input feature (shape = (batch_size, 256))
"""batch_size is equal to the number of sampled paths."""
#x = tf.placeholder(tf.float32, [None, 256])
x = tf.Variable(tf.random_normal([10, 256]))

# number of actions
action_size = 6

# weights and biases
W, b  = _fc_variable([256, action_size])

# unnormalized logits for policy (shape = (batch_size, 6))
logits = tf.matmul(x, W) + b
pi = tf.nn.softmax(logits)
log_pi = tf.log(pi + 1e-20) # 1e-20 to prevent NaN. 


# temperature (get closer to Cartegorical distribution as tau -> 0)
tau = tf.Variable(0.1, name = "temperature")

# sample
y = tf.reshape(gumbel_softmax(logits, tau, hard = False), [-1, action_size])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10):
    print str(i)+"-th sample:\t"+str(sess.run(y))

