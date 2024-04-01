import gym
import rware
env = gym.make("rware-tiny-2ag-v1")
print(env.n_agents)  # 2
print(env.action_space)  # Tuple(Discrete(5), Discrete(5))
print(env.observation_space)  # Tuple(Box(XX,), Box(XX,))

obs = env.reset()  # a tuple of observations

actions = env.action_space.sample()  # the action space can be sampled
print(actions)  # (1, 0)
n_obs, reward, done, info = env.step(actions)

print(done)    # [False, False]
print(reward)  # [0.0, 0.0]
for i in range(10000):
    env.render()
env.close()
