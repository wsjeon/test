import tensorflow as tf

a = tf.Variable(tf.ones([100, 20]))
b = tf.Variable(tf.zeros([100, 20]))

loss1 = tf.losses.mean_squared_error(a, b)
loss2 = tf.reduce_mean(tf.squared_difference(a, b))
loss3 = tf.reduce_mean(tf.pow(a - b, 2))

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  print sess.run(tf.squared_difference(a, b))
  print sess.run(tf.nn.l2_loss(a - b) * 2)
  print sess.run(loss1)
  print sess.run(loss2)
  print sess.run(loss3)
  


