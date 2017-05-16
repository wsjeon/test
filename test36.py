from tensorflow.python.summary.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch

def scalar2arrays(scalarEvents):
  """ converts scalarEvent to set of numpy.array """
  wall_times = []
  steps = []
  values = []
  for event in scalarEvents:
    wall_times.append(event.wall_time)
    steps.append(event.step)
    values.append(event.value)
  
  return np.array(wall_times), np.array(steps), np.array(values)

def find(pattern, path):
  result = []
  for root, dirs, files in os.walk(path):
    for name in files:
      if fnmatch.fnmatch(name, pattern):
        result.append(os.path.join(root, name))
  return result

length = 30000
path_list1 = find("events.*",
    "/home/wsjeon/webdav/MNISTDQN/tmp_plot/maze1/n_step_q_backward_er/1e-05/0/10/1000/200000/{}/".format(length))
path_list2 = find("events.*",
    "/home/wsjeon/webdav/MNISTDQN/tmp_plot/maze1/n_step_q_uniform_er/0.0001/0/10/1000/200000/10/{}/".format(length))
f1 = plt.figure(1)
values1 = []; values2 = []
for i in range(len(path_list1)):
  accumulator1 = EventAccumulator(path_list1[i])
  accumulator2 = EventAccumulator(path_list2[i])
  accumulator1.Reload(); accumulator2.Reload()

  _, step, value = scalar2arrays(accumulator1.Scalars('score'))
  values1.append(np.interp(np.arange(0, length), step, value))

  _, step, value = scalar2arrays(accumulator2.Scalars('score'))
  values2.append(np.interp(np.arange(0, length), step, value))

data1 = np.array(values1); data2 = np.array(values2)
mean1 = np.mean(data1, axis = 0); mean2 = np.mean(data2, axis = 0)
stddev1 = np.std(data1, axis = 0); stddev2 = np.std(data2, axis = 0)
step = np.arange(data1.shape[1])
plt.plot(step, mean1, color = 'blue', linewidth = 1.5)
plt.plot(step, mean2, color = 'red', linewidth = 1.5)
plt.legend(['backward update', 'baseline'])
plt.fill_between(step, mean1 - stddev1, mean1 + stddev1, facecolor = 'blue', alpha = 0.1, edgecolor = 'none')
plt.fill_between(step, mean2 - stddev2, mean2 + stddev2, facecolor = 'red', alpha = 0.1, edgecolor = 'none')
plt.xlabel('time step')
plt.ylabel('cumulative reward')
plt.show()

