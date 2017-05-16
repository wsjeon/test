import tensorflow as tf

a = tf.constant([[1, 2, 3, 4, 5]])

b = a ** 2

with tf.Session() as sess:
  print sess.run(b)

