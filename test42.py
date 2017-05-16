import gym

env = gym.make('PongDeterministic-v3')
env.reset()
cur_lives = env.ale.lives()
for i in range(2000):
  env.render()
  observation, reward, done, info = env.step(env.action_space.sample())
  if cur_lives - env.ale.lives():
    cur_lives = env.ale.lives()
    done = True
  print '{0}\t|{1}\t|{2}\t|{3}'.format(i, reward, done, env.ale.lives())
  if env.ale.lives() == 0 and done:
    env.reset()
    cur_lives = env.ale.lives()
