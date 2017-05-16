from array import array

import numpy as np

import time

f = open("/home/sichoi/maze.txt", 'r')
g = open("/home/sichoi/maze.txt", 'r')

batch_size = 1

for i in range(3):
  x = np.fromfile(f, dtype=np.int8, count = batch_size * 421)
  print x
  x = np.reshape(x, [batch_size, 421]).astype(np.float32)

