import tensorflow as tf

X_idx = tf.reshape(tf.range(10), [-1, 1])

q_idx = tf.reshape(X_idx * tf.ones([5], dtype = tf.int32), [-1])
with tf.Session() as sess:
  print sess.run(X_idx)
  print sess.run(tf.reshape(tf.tile(X_idx, [1, 5]), [-1]))
