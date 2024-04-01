#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/27 22:32
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/27 22:32
# @File         : pipe.py
import gym
import rware
from stable_baselines3 import PPO

env = gym.make('rware-tiny-2ag-v1')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()