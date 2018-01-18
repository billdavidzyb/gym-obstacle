#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
from gym.envs.registration import register
from env import ObstacleEnv
import time

def main():
    register(
            id='Obstacle-v0',
            entry_point=ObstacleEnv,
            max_episode_steps=500,
            reward_threshold=100.0,
        )
    env = gym.make('Obstacle-v0')
    for episode in range(5):
        step_count = 0
        value = 0
        state = env.reset()
        while True:
            env.render()
            if step_count != 60:
                action = 0
            else:
                action = 1
            state, reward, done, info = env.step(action)
            value += reward
            step_count += 1
            if done:
                print('finished episode {}, value = {}'.format(episode, value))
                time.sleep(1)
                break

if __name__ == '__main__':
    main()
