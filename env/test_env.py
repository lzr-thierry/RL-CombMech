import numpy as np
from gym import spaces
import gym
import envs


class TestEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(275,))
        self.action_space = spaces.Box(-1, 1, shape=(275,))
        self.max_steps = 1
        self.current_step = 0

    def step(self, action):
        self.A += action
        self.A = self.A.clip(-1, 1)
        reward = np.sum(- (self.A-0.5)**2) / 10  # 惩罚偏离零

        self.current_step += 1
        done = True

        return self.A.astype(np.float32), reward, done, {}

    def reset(self):
        self.A = np.ones(self.observation_space.shape[0]) * 0.1
        self.current_step = 0
        return self.A.astype(np.float32)


