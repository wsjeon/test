import tensorflow as tf
import time
c = tf.constant('Hello, distributed TensorFlow!')
print c
time.sleep(1)
server = tf.train.Server.create_local_server()
print server
time.sleep(1)
sess = tf.Session(server.target)
print sess
time.sleep(1)
sess.run(c)
