import numpy as np
from gym import spaces
import gym
from wifi_streaming.client_order import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch

class ExoskeletonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device='cpu', host='192.168.4.1', port=8080):
        super(ExoskeletonEnv, self).__init__()
        self.device = device
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.writer = SummaryWriter()
        self.data = np.zeros(9)
        self.current_step = 0
        self.client_socket = self.connect_FREEX(host, port)

    def step(self, action):
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
        data = send_action_to_exoskeleton(self.client_socket, action)
        return observation.squeeze(0).cpu().numpy(), reward, done, info

    def reset(self):
        data = send_action_to_exoskeleton("reset")
        self.data = data
        observation = self.analysis(data)
        return observation.squeeze(0).cpu().numpy()

    def render(self, mode='human', close=False):
        if mode == 'human':
            image = self.generate_state_image()
            self.writer.add_image('Exoskeleton State', image, global_step=self.current_step)
            # 注意：image应该是一个[3, H, W]的Tensor，且数据范围在[0, 1]

    def generate_state_image(self):
        # 待更改成時變率圖
        data = self.data
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        text_str = f"Motor 1 (Right) - Angle: {data[0]} deg, Speed: {data[1]} deg/s, Current: {data[2] * 0.01} A\n" \
                f"Motor 2 (Left) - Angle: {data[3]} deg, Speed: {data[4]} deg/s, Current: {data[5] * 0.01} A\n" \
                f"Roll: {data[6]} deg, Pitch: {data[7]} deg, Yaw: {data[8]} deg"
        ax.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=12)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Pytorch需要tenser值介於[0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image_tensor

    def calculate_reward(self, observation):
        # EMG數值越小越好
        return 0.0

    def check_if_done(self, observation):
        return False

def main():
    # Create the environment instance
    env = ExoskeletonEnv()
    observation = env.reset()
    done = False
    while not done:
        now_step = env.current_step
        sine_angle = np.sin(now_step * 0.1)
        observation, reward, done, _ = env.step(sine_angle, -sine_angle)
        env.render()
    

if __name__ == "__main__":
    main()
