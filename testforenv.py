import matplotlib.pyplot as plt
import keyboard
import asyncio
import random
import asyncio
from matplotlib.animation import FuncAnimation
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
            print("state:")
            print(state)
            await asyncio.sleep(0.05)
    finally:
        env.log_writer.close()

if __name__ == "__main__":
    asyncio.run(main())