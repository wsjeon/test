import tensorflow as tf
import gym

flags = tf.app.flags

flags.DEFINE_string('ENVIRONMENT', 'PongDeterministic-v3', 'environment setting')
flags.DEFINE_integer('task_index', 0, 'one of task indices')

FLAGS = flags.FLAGS

def main():
  ''' Environment setting '''
  envs = [gym.make(FLAGS.ENVIRONMENT) for i in range(2)]

  ''' Agent setting '''
  
  cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
  
  server = tf.train.Server(cluster, job_name = 'local', task_index = FLAGS.task_index)
  
  with tf.device('/job:local/task:0/cpu:0'):
    x = tf.placeholder(tf.float32)
    w = tf.get_variable('w', shape = (), initializer = tf.random_normal_initializer(0.0))
  
  with tf.device('/job:local/task:1/gpu:0'):
    y = w * x
  
  hooks = [tf.train.StopAtStepHook(last_step = 100)]

  global_counter = tf.get_variable('global_step', shape = (),
      initializer = tf.constant_initializer(0), trainable = False)

  assign_op = global_step_tensor.assign(global_counter + 1)
  
  with tf.train.MonitoredTrainingSession(master = server.target,
                                         is_chief = (FLAGS.task_index == 0),
                                         checkpoint_dir = './tmp',
                                         hooks = hooks) as sess:
    while not sess.should_stop():
      print sess.run(y, {x: 1})
      print sess.run(global_counter)
      sess.run(assign_op)
  
if __name__ == '__main__':
  main()
