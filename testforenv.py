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