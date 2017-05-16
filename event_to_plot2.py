from tensorflow.python.summary.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch

def scp_file_to_my_desktop(filepath, savepath = "RL@143.248.49.189:~/experiment1/"):
  bash_command = "sshpass -p rlrl4208 scp "+filepath+" "+savepath  
  import subprocess
  process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
  output, error = process.communicate()

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

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = ax._get_lines.color_cycle.next()
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
  ax.plot(x, y, color=color)
  ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def find(pattern, path):
  result = []
  for root, dirs, files in os.walk(path):
    for name in files:
      if fnmatch.fnmatch(name, pattern):
        result.append(os.path.join(root, name))
  return result

path_list = find("events.*", "/home/wsjeon/experiment1/experiment1_11/")
label = ['0.00025', '0.0005', '0.001', '0.002']
color = ['r', 'g', 'b', 'k']

f1 = plt.figure(1)
for i in range(len(path_list)):
  print "Find event file at ", path_list[i]
  accumulator = EventAccumulator(path_list[i])
  accumulator.Reload()
  wall_times, steps, values = scalar2arrays(accumulator.Scalars('predictron_0/loss/loss_lambda_preturns'))
#  plt.errorbar(steps, values, 0.2)
#  plt.semilogy(steps, values, linewidth = 0.5, c = np.random.rand(3,1))
  plt.semilogy(steps, values, linewidth = 0.25, c = color[i], label = label[i])

plt.xlabel('step (The number of updates)'); plt.ylabel('Mean Squared Error'); plt.axis([0, 60000, 0.0001, 0.1])
plt.legend()
plt.show()
savepath = "/home/wsjeon/experiment1_11_1.pdf"; f1.savefig(savepath); scp_file_to_my_desktop(savepath)


#accumulators = []
#
#accumulator1 =\
#    EventAccumulator('../experiment1/experiment1_5/checkpoint/max_depth_16/learning_rate_0.008/learning_rate_decaying_period_625000/batch_size_800/num_gpus_8/run0/events.out.tfevents.1487701356.GPUSim8')
#accumulator2 =\
#    EventAccumulator('../experiment1/experiment1_1/checkpoint/learning_rate_0.008/learning_rate_decaying_period_625000/batch_size_800/num_gpus_8/run1/events.out.tfevents.1487526369.GPUSim8')
#
#accumulators.append(accumulator1)
#accumulators.append(accumulator2)
#
#f1 = plt.figure(1)
#
#for i in range(len(accumulators)):
#  accumulators[i].Reload()
#
#  wall_times, steps, values = scalar2arrays(accumulators[i].Scalars('predictron_0/loss/loss_lambda_preturns'))
#  
##  plt.semilogy(steps * 800, np.sqrt(values))
#
#  if i == 0:
#    plt.semilogy(steps, values, 'b', linewidth=0.25)
#  elif i == 1:
#    plt.semilogy(steps, values /2 * 20, 'r', linewidth=0.25)
#
#plt.xlabel('step (The number of updates)'); plt.ylabel('Mean Squared Error'); plt.axis([0, 120000, 0.0001, 0.1])
#savepath = "/home/wsjeon/experiment1_5_4.pdf"; f1.savefig(savepath); scp_file_to_my_desktop(savepath)
#
#f2 = plt.figure(2)
#_, steps, values = scalar2arrays(accumulators[0].Scalars('predictron_0/loss/loss_lambda_preturns'))
#plt.semilogy(steps, values, 'b', linewidth=0.25)
#
#plt.xlabel('step (The number of updates)'); plt.ylabel('Mean Squared Error'); plt.axis([0, 300000, 0.0001, 0.1])
#savepath = "/home/wsjeon/experiment1_5_5.pdf"; f2.savefig(savepath); scp_file_to_my_desktop(savepath)
