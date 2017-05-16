import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

x = tf.get_variable('x', shape = (), initializer = tf.constant_initializer(1), dtype = tf.float32)

def iter_fun(i, y):
  y = x * layers.batch_norm(y)
  return (i + 1, y)

_, result = tf.while_loop(
      cond = lambda i, *_: i < 5,
      body = iter_fun,
      loop_vars = (0, tf.constant([1, 2, 3, 4], dtype = tf.float32)))

loss = tf.reduce_mean(result)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops = tf.group(*update_ops)

#with tf.control_dependencies(update_ops):
#  opt = tf.train.AdamOptimizer().minimize(loss)
opt = tf.train.AdamOptimizer().minimize(loss) # - !

  

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(update_ops) # - !
  sess.run(opt)
