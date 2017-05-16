import tensorflow as tf
import numpy as np
params = tf.reshape(tf.range(2 * 10 * 8 * 8), [2, 10, 8, 8])
x0 = 1; y0 = 1
x1 = 1; y1 = 2
indices = tf.Variable(
[
[[0, 0, x0, y0],
 [0, 1, x0, y0],
 [0, 2, x0, y0],
 [0, 3, x0, y0],
 [0, 4, x0, y0],
 [0, 5, x0, y0],
 [0, 6, x0, y0],
 [0, 7, x0, y0],
 [0, 8, x0, y0],
 [0, 9, x0, y0]],
[[0, 0, x1, y1],
 [0, 1, x1, y1],
 [0, 2, x1, y1],
 [0, 3, x1, y1],
 [0, 4, x1, y1],
 [0, 5, x1, y1],
 [0, 6, x1, y1],
 [0, 7, x1, y1],
 [0, 8, x1, y1],
 [0, 9, x1, y1]]
]
)
output = tf.gather_nd(params, indices)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(output)
  
