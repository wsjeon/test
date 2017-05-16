import tensorflow as tf
import numpy as np

param = tf.constant(np.random.random((2,3,4,2,4)), dtype = tf.float32)
#transpose = tf.transpose(param, [0, 2, 1])
action = tf.constant(np.random.randint(4, size = (2, 1)))
one_hot = tf.one_hot(action, 4)
one_hot = tf.reshape(one_hot, [2, 1, 1, 1, 4])
mul = param * one_hot

with tf.Session() as sess:
  print sess.run(param)
#  print sess.run(transpose).shape
  print sess.run(action)
  print sess.run(one_hot)
  print sess.run(mul)
  print sess.run(tf.reduce_sum(mul, axis = 4))
