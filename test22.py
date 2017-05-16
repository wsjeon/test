from tensorflow.python.summary.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

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

accumulators = []

accumulator1 = EventAccumulator('events.out.tfevents.1487526369.GPUSim8')
accumulator2 = EventAccumulator('events.out.tfevents.1487511530.GPUSim10')

accumulators.append(accumulator1)
accumulators.append(accumulator2)

for i in range(len(accumulators)):
  accumulators[i].Reload()

  wall_times, steps, values = scalar2arrays(accumulators[i].Scalars('predictron_0/loss/loss_lambda_preturns'))
  
  plt.semilogy(steps * 800, np.sqrt(values))
  
  plt.show()
