import tensorflow as tf
import numpy as np

S = tf.constant(np.random.random((5, 4)), dtype = tf.float32)
A = tf.constant(np.array([np.random.choice(4, size = 5)]).reshape([-1]))
E = tf.nn.embedding_lookup(S, A)
O = tf.one_hot(A, 4, dtype = tf.float32)
R = O * S
RS = tf.reduce_sum(R, axis = 1)
with tf.Session() as sess:
  print sess.run(S)
  print sess.run(A)
  print sess.run(E)
  print sess.run(O)
  print sess.run(R)
  print sess.run(RS)
