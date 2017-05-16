import tensorflow as tf
import numpy as np

x_data = np.array([
    [1., 1., 1., 1., 1.],
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]]).astype(float)

y_data = np.array([
    1., 2., 3., 4., 5.
]).astype(float)

print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, [3,5], name="X")
Y = tf.placeholder(tf.float32, [5], name="Y")

W = tf.Variable(tf.zeros([3, 3], dtype=tf.float32), name="W")
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# (3,3),(3,5)
H = tf.matmul(W,X)+b

cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(np.array(x_data).shape)
print(np.array(y_data).shape)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}), sess.run(W))
