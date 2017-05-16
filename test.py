import tensorflow as tf 
import numpy as np

class TT(object):
  def __init__(self):

    self.a = 10
  def tt(self):
    def _hello(t):
      return t + self.a
    print _hello(10)


TTT = TT()

TTT.tt()

a = tf.Variable(tf.random_uniform([10, 1]))
b = tf.Variable(tf.random_uniform([10, 1]))
c = a + b
d = a * b
e = a ** 2
f = tf.random_uniform([1])

print c
print d
print e
print (f ** 3) * a
g= tf.transpose(c)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(a) 
  print sess.run(tf.reshape(a, [5, 2]))
  
