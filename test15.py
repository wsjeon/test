import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# data
len_of_data = 100000
batch_size1 = 100 
batch_size2 = 800

x_data = 5 * np.random.uniform(size = (len_of_data, 1))
n_data = np.random.randn(len_of_data, 1)
y_data = x_data + n_data

# model 1
X1_= tf.placeholder(tf.float32, [None, 1])
Y1_= tf.placeholder(tf.float32, [None, 1])
A1 = tf.Variable(0.0)
B1 = tf.Variable(0.0)
Y1 = A1 * X1_ + B1
loss1 = tf.reduce_mean(tf.squared_difference(Y1_, Y1))
opt1 = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss1, var_list = [A1, B1])

# model 2
X2_= tf.placeholder(tf.float32, [None, 1])
Y2_= tf.placeholder(tf.float32, [None, 1])
A2 = tf.Variable(0.0)
B2 = tf.Variable(0.0)
Y2 = A2 * X2_ + B2
loss2 = tf.reduce_mean(tf.squared_difference(Y2_,- Y2))
opt2 = tf.train.AdamOptimizer(learning_rate = 0.008, beta1 = 0.43046721, beta2 = 0.992027944).minimize(loss2, var_list = [A2, B2])

loss1_list = []
loss2_list = []

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(len_of_data): # step sync
    if step % batch_size1 == 0:
      t = step / batch_size1
      xs = x_data[t * batch_size1:(t + 1) * batch_size1, :]
      ys = y_data[t * batch_size1:(t + 1) * batch_size1, :]
      _, loss, a, b = sess.run([opt1, loss1, A1, B1], feed_dict = {X1_: xs, Y1_:ys})
      print a
      loss1_list.append(loss)
           
    if step % batch_size2 == 0:
      t = step / batch_size2
      xs = x_data[t * batch_size2:(t + 1) * batch_size2, :]
      ys = y_data[t * batch_size2:(t + 1) * batch_size2, :]
      _, loss, a, b = sess.run([opt2, loss2, A2, B2], feed_dict = {X2_: xs, Y2_:ys})
      loss2_list.append(loss)

loss_list = loss1_list + loss2_list
loss_list = np.array(loss_list)
loss1_list = np.array(loss1_list)
loss2_list = np.array(loss2_list)


plt.figure(1)

plt.plot(np.arange(0, len_of_data, batch_size1), loss1_list, 'r', linewidth = 0.25, label = '100')
plt.plot(np.arange(0, len_of_data, batch_size2), loss2_list, 'b', linewidth = 0.5, label = '800')
plt.axis([0, len_of_data, 0, np.amax(loss_list)])
plt.xlabel('The number of samples')
plt.ylabel('Loss')
plt.legend()

plt.show()
