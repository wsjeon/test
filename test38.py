import tensorflow as tf
import numpy as np
import os

#with tf.variable_scope('online_net'):
with tf.variable_scope('sc1'):
  a = tf.get_variable('a', [1], initializer = tf.constant_initializer(3.0))
with tf.variable_scope('sc2'):
  b = tf.get_variable('b', [1], initializer = tf.constant_initializer(3.0))
sc = tf.get_variable_scope().name
c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = os.path.join(sc, 'sc2'))
print c

print tf.get_variable_scope().name



