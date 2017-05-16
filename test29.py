import tensorflow as tf
import numpy as np

a = tf.Variable(tf.random_normal([3, 3]))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(a)
  print sess.run(tf.argmax(a, axis = 0))
