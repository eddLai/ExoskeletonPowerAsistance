from wifi_streaming import client_order
from EMG import emgdata
import asyncio
import gym
from gym import spaces
import numpy as np
from tensorboardX import SummaryWriter

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu',save_path="runs/Env_test" , host='192.168.4.1', url= "ws://localhost:31278/ws", port=8080):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.host = host
        self.port = port
        self.uri = url
        self.observation = np.zeros(9)
        self.emg_observation = np.zeros(6)
        self.bp_parameter = np.zeros((6,8))
        self.nt_parameter = np.zeros((6,2))
        self.lp_parameter = np.zeros((6,4))
        self.initial_max_min_rms_values = np.zeros((6,2))
        self.current_step = 0
        self.init_time = 0
        self.reward = 0
        # Initialize reader and writer as None; they will be set in reset
        self.reader = None
        self.writer = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.log_writer = SummaryWriter(save_path)
        
    async def step(self, action):
        return await asyncio.run(self.async_step(action))
    async def async_step(self, action):
        # 改回用send_action_to_exoskeleton_speed函數
        # await client_order.FREEX_CMD(self.writer, "C", action[0], "C", action[1])
        new_observation, new_emg_observation, new_bp_parameter, new_nt_parameter, new_lp_parameter = await client_order.get_INFO(self.reader, self.uri ,self.bp_parameter, self.nt_parameter, self.lp_parameter)
        
        if not np.all(new_observation==0):
            print("data is good")
            self.observation = new_observation
        else:
            print("use the old data")
        if not np.all(new_emg_observation==0):
            self.emg_observation = np.sqrt(np.mean(new_emg_observation**2, axis=1))
            self.bp_parameter = new_bp_parameter
            self.nt_parameter = new_nt_parameter
            self.lp_parameter = new_lp_parameter
            if self.init_time <= 10000:
                self.init_time = self.init_time + 50  #len(new_emg_observation)
        
        await client_order.send_action_to_exoskeleton(self.writer, action, self.observation ,"speed")
        self.reward = await self.calculate_reward()
        done = self.check_if_done(self.observation)
        self.current_step += 1
        self.render()
        return np.concatenate([self.observation, self.emg_observation], axis=0), self.reward, done, {}
    
    def reset(self):
        return asyncio.run(self.async_reset())
    async def async_reset(self):
        # Properly handle async connection in reset
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader, self.writer = await client_order.connect_FREEX(self.host, self.port)
        await client_order.FREEX_CMD(self.writer, "E", "0", "E", "0")
        self.observation, self.emg_observation, self.bp_parameter, self.nt_parameter, self.lp_parameter = await client_order.get_INFO(self.reader, self.uri ,self.bp_parameter, self.nt_parameter, self.lp_parameter)
        self.emg_observation = np.sqrt(np.mean(self.emg_observation**2, axis=1))
        return np.concatenate([self.observation, self.emg_observation], axis=0)  #self.emg_observation的格式
        # return np.zeros(15)

    async def calculate_reward(self):
        reward, self.initial_max_min_rms_values = await emgdata.calculate_emg_level(self.emg_observation, self.initial_max_min_rms_values, self.init_time)
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
        for i in range(self.emg_observation.shape[0]):
            self.log_writer.add_scalar(f'EMG/Channel_{i+1}', self.emg_observation[i], self.current_step)
