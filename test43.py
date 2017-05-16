import numpy as np
import time

start = time.time()
a = np.empty((0))
for i in range(100000):
  a = np.append(a, np.array([i]))
finish = time.time()

print finish - start

start = time.time()
a = []
for i in range(100000):
  a.append(np.array([i]))
finish = time.time()

print finish - start

start = time.time()
a = np.empty((0))
for i in range(100000):
  a.resize(len(a) + 1)
  a[-1] = i
finish = time.time()

print finish - start
