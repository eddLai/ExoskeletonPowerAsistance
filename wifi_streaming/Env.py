from wifi_streaming import client_order
from EMG import emg_nonasync
import asyncio
import gym
from gym import spaces
import numpy as np
from tensorboardX import SummaryWriter
import time
import keyboard

channel_names = [
    'Tibialis_anterior_right',  # 通道1: 右腿脛前肌
    'Rectus Femoris_right',     # 通道2: 右腿股直肌
    'Biceps_femoris_right',     # 通道3: 右腿股二頭肌
    'Gastrocnemius_right',      # 通道4: 右腿腓腸肌
    'Tibialis_anterior_left',   # 通道5: 左腿脛前肌
    'Rectus Femoris_left',      # 通道6: 左腿股直肌
    'Biceps_femoris_left',      # 通道7: 左腿股二頭肌
    'Gastrocnemius_left'        # 通道8: 左腿腓腸肌
]


class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, log_writer , device='cpu', host='192.168.4.1', url= "ws://localhost:31278/ws", port=8080):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.host = host
        self.port = port
        self.uri = url
        self.observation = np.zeros(9)
        self.emg_observation = np.zeros(8)
        self.filtered_emg_observation = np.zeros((8,50))
        self.bp_parameter = np.zeros((8,8))
        self.nt_parameter = np.zeros((8,2))
        self.lp_parameter = np.zeros((8,4))
        self.initial_max_min_rms_values = np.zeros((8,2))
        self.current_step = 0
        self.init_time = 0
        self.reward = 0
        self.sock = client_order.connect_FREEX(self.host, self.port)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.log_writer = log_writer
    
    def step(self, action):
        # 改回用send_action_to_exoskeleton_speed函數
        self.observation, self.filtered_emg_observation, self.bp_parameter, self.nt_parameter, self.lp_parameter = client_order.get_INFO(self.sock, self.uri ,self.bp_parameter, self.nt_parameter, self.lp_parameter)
        #window.update_plot(self.filtered_emg_observation[0])
        self.emg_observation = np.sqrt(np.mean(self.filtered_emg_observation**2, axis=1))

        client_order.send_action_to_exoskeleton(self.sock, action, self.observation ,"speed")
        self.reward = self.calculate_reward()
        done = self.check_if_done(self.observation)
        self.current_step += 1
        self.render()
        return np.concatenate([self.observation, self.emg_observation], axis=0), self.reward, done, {}
    
    def reset(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        print("disconnect")
        self.sock= client_order.connect_FREEX(self.host, self.port)
        print("re-connected")
        time.sleep(2)
        client_order.FREEX_CMD(self.sock, "A", "0000", "A", "0000")
        print("reset to angle, be relaxed")
        time.sleep(5)
        client_order.FREEX_CMD(self.sock, "E", "0", "E", "0")
        input("Press Enter to Reset Muscle Power Level")
        self.emg_observation = np.zeros(8)
        self.filtered_emg_observation = np.zeros((8,50))
        self.bp_parameter = np.zeros((8,8))
        self.nt_parameter = np.zeros((8,2))
        self.lp_parameter = np.zeros((8,4))
        self.initial_max_min_rms_values = np.zeros((8,2))
        self.init_time = 0
        print("Please walk naturally for 10 seconds.")
        while self.init_time <= 10000:
            self.init_time = self.init_time + 50  #len(new_emg_observation)
            self.observation, self.filtered_emg_observation, self.bp_parameter, self.nt_parameter, self.lp_parameter = client_order.get_INFO(self.sock, self.uri ,self.bp_parameter, self.nt_parameter, self.lp_parameter)
            #window.update_plot(self.filtered_emg_observation[0])
            self.emg_observation = np.sqrt(np.mean(self.filtered_emg_observation**2, axis=1))
            self.calculate_reward()
            if self.init_time % 1000 == 0:
                print("Countdown: ",10 - int(round(self.init_time/1000)))
        print("first data recv")
        return np.concatenate([self.observation, self.emg_observation], axis=0)  #self.emg_observation的格式
        # return np.zeros(15)

    def calculate_reward(self):
        reward, self.initial_max_min_rms_values = emg_nonasync.calculate_emg_level(self.emg_observation, self.initial_max_min_rms_values, self.init_time)
        return reward

    def check_if_done(self, observation):
        # Implement logic to check if the episode is done
        return False
    
    def render(self, mode='human', close=False):
        self.log_writer.add_scalars('Joint/Angle', {'Joint1': self.observation[0], 'Joint2': self.observation[3]}, self.current_step)
        self.log_writer.add_scalars('Joint/Velocity', {'Joint1': self.observation[1], 'Joint2': self.observation[4]}, self.current_step)
        self.log_writer.add_scalars('Joint/Current', {'Joint1': self.observation[2], 'Joint2': self.observation[5]}, self.current_step)
        self.log_writer.add_scalars('IMU', {'Roll': self.observation[6], 'Pitch': self.observation[7], 'Yaw':self.observation[8]}, self.current_step)
        self.log_writer.add_scalar('Reward', self.reward, self.current_step)
        filtered_emg_step = self.current_step*50
        for i in range(self.emg_observation.shape[0]):
            for j in range(50):
                self.log_writer.add_scalar(f'Filtered_EMG/{channel_names[i]}', self.filtered_emg_observation[i][j], filtered_emg_step+j)
            self.log_writer.add_scalar(f'sqrted EMG/Channel_{channel_names[i]}', self.emg_observation[i], self.current_step)
