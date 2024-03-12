from wifi_streaming import client_order
import asyncio
import gym
from gym import spaces
import numpy as np
from tensorboardX import SummaryWriter

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu', name='test', host='192.168.4.1', port=8080):
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
        self.log_writer = SummaryWriter("runs/{}".format(name))

    def step(self, action):
        return asyncio.run(self.async_step(action))

    async def async_step(self, action):
        # 改回用send_action_to_exoskeleton_speed函數
        await client_order.FREEX_CMD(self.writer, "C", action[0], "C", action[1])
        new_observation = await client_order.get_INFO(self.reader)
        # new_observation = np.random.rand(9)
        if new_observation.shape[0] != 0:
            self.observation = new_observation
        self.reward = self.calculate_reward(self.observation)
        done = self.check_if_done(self.observation)
        self.current_step += 1
        self.render()
        return self.observation, self.reward, done, {}

    def reset(self):
        return asyncio.run(self.async_reset())

    async def async_reset(self):
        # Properly handle async connection in reset
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader, self.writer = await client_order.connect_FREEX(self.host, self.port)
        self.observation = await client_order.get_INFO(self.reader)
        # new_observation = np.random.rand(9)
        return self.observation

    def calculate_reward(self, observation):
        # Implement reward calculation
        return 0.0

    def check_if_done(self, observation):
        # Implement logic to check if the episode is done
        return False
    
    def render(self, mode='human', close=False):
        self.log_writer.add_scalars('Joint/Angle', {'Joint1': self.observation[0], 'Joint2': self.observation[3]}, self.current_step)
        self.log_writer.add_scalars('Joint/Velocity', {'Joint1': self.observation[1], 'Joint2': self.observation[4]}, self.current_step)
        self.log_writer.add_scalars('Joint/Acceleration', {'Joint1': self.observation[2], 'Joint2': self.observation[5]}, self.current_step)
        self.log_writer.add_scalars('IMU', {'Roll': self.observation[6], 'Pitch': self.observation[7], 'Yaw':self.observation[8]}, self.current_step)
        self.log_writer.add_scalar('Reward', self.reward, self.current_step)