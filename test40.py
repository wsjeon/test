import tensorflow as tf
import numpy as np

a = tf.Variable([1, 2, 3])

b = a[-1]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(b)
