import tensorflow as tf
import numpy as np

z = 2 * tf.ones([100])
T = tf.nn.l2_loss(z)


with tf.Session() as sess:
  print sess.run(z)
  print sess.run(T)
