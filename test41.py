import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.slim import conv2d

init = tf.constant_initializer(np.zeros((1, 3, 3, 1)))

def net():
  net = tf.get_variable('a', shape = (1, 3, 3, 1), initializer = init)
  net = conv2d(net, 32, [2, 2], stride = 1, scope = 'conv')
  print net
  return net

net_template = tf.make_template('net', net)

with tf.device('/cpu:0'):
  with tf.variable_scope('cpu'):
    net = net_template()

with tf.device('/gpu:0'):
  with tf.variable_scope('gpu'):
    net = net_template()

cpu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cpu')
gpu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'gpu')

for var in cpu_vars: 
  print var
for var in gpu_vars:
  print var


with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
  pass

