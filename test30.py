import tensorflow as tf
import numpy as np

# q = tf.Variable(tf.random_normal([4, 2, 2, 3]))
# a = tf.transpose(q, perm = [0, 3, 1, 2])
# b = tf.expand_dims(a, axis = 1)
# c = tf.concat([b] * 5, axis = 1)
# d = tf.reshape(c, [-1, 3, 2, 2])
# 
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
# 
#   print sess.run(q).shape
#   print sess.run(a).shape
#   print sess.run(b).shape
#   print sess.run(c).shape
#   print sess.run(d).shape
# 
#   D = sess.run(d)
#   print D[0, :, :, :] == D[4, :, :, :]
# 
q = np.random.randn(4, 2, 2, 3)
#a = np.transpose(q, (0, 3, 1, 2))
a = np.expand_dims(q, axis = 1)
b = np.concatenate([a] * 5, axis = 1)
#d = np.reshape(c, [-1, 2, 2])

print q.shape, a.shape, b.shape
S1 = np.random.choice(2, size = (4, 5))
S2 = np.random.choice(2, size = (4, 5))
print S1, S2

S1 = np.reshape(S1, [-1])
S2 = np.reshape(S2, [-1])
