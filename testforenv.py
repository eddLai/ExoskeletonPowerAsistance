import matplotlib.pyplot as plt
import keyboard
import asyncio
import random
import asyncio
from matplotlib.animation import FuncAnimation
from wifi_streaming import Env

async def check_if_safe(limit:int, angle, speed):
    print(angle)
    angle = int(angle)
    if angle is None:
        return 0
    elif (angle >= limit and speed > 0) or (angle <= -limit and speed < 0):
        return 0
    else:
        return speed

async def main():
    try:
        env = Env.ExoskeletonEnv2(device='cuda', save_path="runs/testforenv")
        state = await env.async_reset()
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            action1 = random.randint(-5, 5)
            print("motor R: ", action1, "\tangle: ", state[0])
            action1 = await check_if_safe(10, state[0], action1)
            print("modified motor R: ", action1)
            action2 = random.randint(-5, 5)
            print("motor L: ", action2, "\tangle: ", state[3])
            action2 = await check_if_safe(10, state[3], action2)
            print("modified motor L: ", action2)
            print("----------------------------")
            state, reward, done, info = await env.async_step([action1,action2])
            await asyncio.sleep(0.05)
    finally:
        if not asyncio.get_running_loop().is_closed():
            env.log_writer.close()
        env.log_writer.close()

if __name__ == "__main__":
    asyncio.run(main())