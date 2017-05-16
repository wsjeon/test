from maze import MazeGenerator
import time
import numpy as np

f = open("maze.txt", "wb")

for i in range(100000):
  start_time = time.time()
  maze_gen = MazeGenerator(height = 20, width = 20, density = 0.3)
  mazes, labels = maze_gen.generate_labelled_mazes(1)
#  print np.array(mazes).reshape(-1).tolist()+np.array(labels).reshape(-1).tolist()
  single_maze = np.array(mazes).reshape(-1).tolist()+np.array(labels).reshape(-1).tolist()
  print single_maze
  print bin(single_maze[0])
  
  
#  print len(labels)
#
#for i in range(100000):
#	start_time=time.time()
#	maze_gen = MazeGenerator(height = 20, width = 20, density = 0.3)
#	maze_ims, maze_labels = maze_gen.generate_labelled_mazes(100)
#  print maze_ims
#  a[i]= time.time()-start_time
#
#for i in range(10):
#	print "saving"
#	print time.time()-ac
#	sio.savemat(("gen_im_26_%d.mat" % i),{'im':maze_ims[10000*i:10000*(i+1),:,:,:,:]})
#	sio.savemat(("gen_lb_26_%d.mat" % i),{'lb':maze_labels[10000*i:10000*(i+1),:,:]})
#print np.mean(a)
#print time.time()-ac
