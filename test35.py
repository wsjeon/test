import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 15, 31)
data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
mean = np.mean(data, axis = 0)
stddev = np.std(data,axis=0)
step = np.arange(data.shape[1])
plt.plot(step, mean, color = 'blue', linewidth = 1.5)
plt.fill_between(step, mean - stddev, mean + stddev, facecolor = 'blue', alpha = 0.1,
    edgecolor = 'none')
plt.legend(['Single Goal'])
plt.xlabel('number of episodes')
plt.ylabel('average accuracy')
plt.show()

