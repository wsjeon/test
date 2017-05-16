import tensorflow as tf
import numpy as np
import threading

q = tf.FIFOQueue(100, tf.float32)
init_q = q.enqueue_many(np.arange(100))
get_output = q.dequeue()
get_new_input = get_output + 1
add_q = q.enqueue(get_new_input)

with tf.Session() as sess:
  sess.run(init_q)
  for i in range(300):
    print sess.run(add_q)

  for i in range(100):
    print sess.run(get_output)

