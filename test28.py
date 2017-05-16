import tensorflow as tf
import numpy as np

x = tf.Variable(3.0)

y = x

opt = tf.train.GradientDescentOptimizer(0.01).minimize(y)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(opt)
  print sess.run(y)
