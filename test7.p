import tensorflow as tf
import tensorflow.contrib.slim as slim

def mapping(feat, is_training=True, reuse=False):
  batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
  def _network():
    net = slim.fully_connected(feat, 16,
        activation_fn = tf.nn.sigmoid,
        weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
        normalizer_fn = slim.batch_norm,
        normalizer_params = batch_norm_params,
        scope='m_fc1')
    net = slim.dropout(net, keep_prob=0.7, is_training=is_training, scope='m_dr')  
    out = slim.fully_connected(net, 2,
        activation_fn = None,
        normalizer_fn = None,
        scope='m_fc2')
    return out
 
  with tf.variable_scope("mapping") as scope:   
    try:
      return _network()
    except ValueError:
      scope.reuse_variables()
      return _network()

# network
feat = tf.random_uniform([2,2])
out1 = mapping(feat)
out2 = mapping(feat)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print tf.trainable_variables()
  print out1, out2
  
