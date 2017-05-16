import tensorflow as tf
import numpy as np

x = tf.Variable([5.0])
x_ = tf.placeholder_with_default(x, [None])
y = x_ ** 2

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(y)
  print sess.run(y, feed_dict = {x_: [20.0]})
  print sess.run(tf.gradients(y, [x_]))
  print sess.run(tf.gradients(y, [x_]), feed_dict = {x_: [20.0]})
