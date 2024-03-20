import matplotlib.pyplot as plt
import keyboard
import asyncio
from matplotlib.animation import FuncAnimation
from wifi_streaming import Env, client_order
import numpy as np

async def main():
    try:
        env = Env.ExoskeletonEnv(device='cuda', save_path="runs/testforenv")
        state = await env.async_reset()
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            # action1 = np.random.uniform(-1, 1)
            action1 = 0
            action2 = np.random.uniform(-1, 1)
            state, reward, done, info = await env.async_step([action1,action2])
            await asyncio.sleep(0.05)
    finally:
        if not asyncio.get_running_loop().is_closed():
            await client_order.FREEX_CMD(env.writer, "E", "0", "E", "0")
            env.log_writer.close()
        await client_order.FREEX_CMD(env.writer, "E", "0", "E", "0")
        env.log_writer.close()

if __name__ == "__main__":
    asyncio.run(main())