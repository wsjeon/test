import tensorflow as tf 
import numpy as np

x = tf.placeholder(tf.float32)

y1 = x ** 2

y2 = tf.Variable(1.0)

t = tf.placeholder_with_default(1, ())
def f1(): return y1 ** 2
def f2(): return y2
z = tf.cond(tf.less(0, t), f1, f2)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(y1, feed_dict = {x: 2.0})
  print sess.run(z, feed_dict = {x: 2.0})
  print sess.run(tf.gradients(z, x), feed_dict = {x: 2.0})
  print sess.run(z, feed_dict = {t: 0, x: 2.0})



