import numpy as np
from gym import spaces
import gym
from wifi_streaming import client_order
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
import keyboard

'''EXO ENVIRONMENT中統一用observation, 外部統一用state'''

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu', host='192.168.4.1', port=8080):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.data = np.zeros(9)
        self.current_step = 0
        self.client_socket = client_order.connect_FREEX(host, port)
        # self.writer = SummaryWriter("runs/ExoENVtest")

    def step(self, action, type):
        raw_data = self.get_INFO()
        if raw_data:
            data = self.analysis(raw_data)
            self.data = data if data else self.data
        else:
            print("Failed to get data.") 
        observation = self.data
        reward = self.calculate_reward(observation)
        done = self.check_if_done(observation)
        info = {}
        data = client_order.send_action_to_exoskeleton(self.client_socket, action, type)
        return observation.squeeze(0).cpu().numpy(), reward, done, info

    def reset(self):
        data = client_order.send_action_to_exoskeleton(self.client_socket, "reset")
        self.data = data
        observation = client_order.analysis(data)
        return observation.squeeze(0).cpu().numpy()

    # def render(self, mode='human', close=False):
    #     self.writer.add_scalars('Joint/Angle', {'Joint1': self.data[0], 'Joint2': self.data[3]}, self.current_step)
    #     self.writer.add_scalars('Joint/Velocity', {'Joint1': self.data[1], 'Joint2': self.data[4]}, self.current_step)
    #     self.writer.add_scalars('Joint/Acceleration', {'Joint1': self.data[2], 'Joint2': self.data[5]}, self.current_step)
    #     self.writer.add_scalars('IMU', {'Roll': self.data[6], 'Pitch': self.data[7], 'Yaw':self.data[8]}, self.current_step)
    #     self.writer.add_scalar('Reward', reward, self.current_step)

    def calculate_reward(self, observation):
        # EMG數值越小越好
        return 0.0

    def check_if_done(self, observation):
        return False
    

if __name__ == "__main__":
    env = ExoskeletonEnv(device='cuda')
    state = env.reset()
    done = False
    while not done:
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break
        now_step = env.current_step
        sine_angle = np.sin(now_step * 0.1)
        state, reward, done, _ = env.step([sine_angle, -sine_angle], 'angle')
        # env.render()