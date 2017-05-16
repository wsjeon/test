from array import array

import numpy as np

import time

f = open("/home/sichoi/maze.txt", 'r')
g = open("/home/sichoi/maze.txt", 'r')

batch_size = 100

at = time.time()
ClassArray = array('b')
ClassArray.fromfile(f, batch_size * 421)
x = np.reshape(ClassArray, [batch_size, 421])
bt = time.time()
print bt - at
print x

#at = time.time()
#ClassArray = array('b')
#ClassArray.fromfile(f, batch_size * 421)
#x = np.reshape(ClassArray, [batch_size, 421])
#bt = time.time()
#print bt - at
#print x

ct = time.time()
x = np.fromfile(g, dtype=np.int8, count = batch_size * 421)
x = np.reshape(x, [batch_size, 421]).astype(np.float32)
dt = time.time()
print dt - ct
print x

print (bt - at) / (dt - ct)

#at = time.time()
#x = np.fromfile(g, dtype=np.int8, count = batch_size * 421)
#x = np.reshape(x, [batch_size, 421])
#bt = time.time()
#print bt - at
#print x


