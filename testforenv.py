import numpy as np
from gym import spaces
import gym
from wifi_streaming import client_order
from EMG import emgdata
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
import keyboard
import time
import asyncio
import random
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import asyncio
import aiohttp
import websockets
import json
from matplotlib.animation import FuncAnimation
import pandas as pd
from wifi_streaming import Env

'''EXO ENVIRONMENT中統一用observation, 外部統一用state'''

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu', host='192.168.4.1', port=8080):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.host = host
        self.port = port
        self.observation = np.zeros(9)
        self.current_step = 0
        self.reward = 0
        # Initialize reader and writer as None; they will be set in reset
        self.reader = None
        self.writer = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.log_writer = SummaryWriter("runs/Env_test")
        # Initialize emg process
        self.uri = "ws://localhost:31278/ws"
        self.emg_observation = np.zeros((6,50))
        self.ft_parameter = np.zeros((6,3))
        self.initial_max_min_rms_values = np.zeros((6,2))
        self.times = 0
        
    async def step(self, action):
        # 改回用send_action_to_exoskeleton_speed函數
        await client_order.FREEX_CMD(self.writer, "C", action[0], "C", action[1])
        new_observation, new_emg_observation, new_ft_parameter = await client_order.get_INFO(self.reader,self.uri ,self.ft_parameter)
        
        if new_observation.shape[0] != 0:
            self.observation = new_observation
        if new_emg_observation.shape[0] != 0:
            self.emg_observation = new_emg_observation
            self.ft_parameter = new_ft_parameter
            if self.times <= 10000:
                self.times = self.times + 50  #len(new_emg_observation)
        self.reward = self.calculate_reward()
        done = self.check_if_done(self.observation)
        self.current_step += 1
        self.render()
        return self.observation, self.reward, done, {}

    async def reset(self):
        # Properly handle async connection in reset
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader, self.writer = await client_order.connect_FREEX(self.host, self.port)
        self.observation = await client_order.get_INFO(self.reader)

        return self.observation

    async def calculate_reward(self):
        # Implement reward calculation
        
        reward, self.initial_max_min_rms_values = await emgdata.calculate_emg_level(self.emg_observation, self.initial_max_min_rms_values, self.times)
        
        return reward

    def check_if_done(self, observation):
        # Implement logic to check if the episode is done
        return False
    
    def render(self, mode='human', close=False):
        self.log_writer.add_scalars('Joint/Angle', {'Joint1': self.observation[0], 'Joint2': self.observation[3]}, self.current_step)
        self.log_writer.add_scalars('Joint/Velocity', {'Joint1': self.observation[1], 'Joint2': self.observation[4]}, self.current_step)
        self.log_writer.add_scalars('Joint/Acceleration', {'Joint1': self.observation[2], 'Joint2': self.observation[5]}, self.current_step)
        self.log_writer.add_scalars('IMU', {'Roll': self.observation[6], 'Pitch': self.observation[7], 'Yaw':self.observation[8]}, self.current_step)
        self.log_writer.add_scalar('Reward', self.reward, self.current_step)
    
async def main():
    try:
        env = Env.ExoskeletonEnv2(device='cuda', save_path="runs/testforenv")
        await env.async_reset()
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            # action = env.action_space.sample()
            action1 = str(random.randint(-5, 5) * 1000)
            action2 = str(random.randint(-5, 5) * 1000)
            state, reward, done, info = await env.async_step([action1,action2])
            print(state)
            await asyncio.sleep(0.05)
    finally:
        pass
        # env.writer.close()
        # await env.writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
    # env = ExoskeletonEnv(device='cuda')
    # done = False
    # while not done:
    #     if keyboard.is_pressed('q'):
    #         print("Exiting...")
    #         break
    #     # now_step = env.current_step
    #     # # np.sin(now_step)*5/18
    #     sine_speed = np.array([0, 0])
    #     state, reward, done, _ = env.step(sine_speed, 'speed')
    #     print(state)
        # env.render()