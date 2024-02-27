from client_order import *
import gym
from gym import spaces
import numpy as np

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu'):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def step(self, action):
        data = send_action_to_exoskeleton(action)
        observation = self.analysis(data)
        reward = self.calculate_reward(observation)
        done = self.check_if_done(observation)
        info = {}
        return observation.squeeze(0).cpu().numpy(), reward, done, info

    def reset(self):
        data = send_action_to_exoskeleton("reset")
        observation = self.analysis(data)
        return observation.squeeze(0).cpu().numpy()

    def render(self, mode='human', close=False):
        pass

    def calculate_reward(self, observation):
        # 根据观测值计算奖励
        # 这需要你根据具体任务来定义
        return 0.0