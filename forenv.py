import keyboard
from matplotlib.animation import FuncAnimation
from tensorboardX import SummaryWriter
from wifi_streaming import Env, client_order
import time
import numpy as np

def main():
    writer = SummaryWriter("runs/recording_EXO_newEMG_plot")
    env = Env.ExoskeletonEnv(writer, device='cuda')
    try:
<<<<<<< HEAD
        writer = SummaryWriter("runs/recording_EXO_newEMG_plot")
        env = Env.ExoskeletonEnv(writer, device='cuda')
        state = env.reset(is_recording=True)
=======
        state = env.reset(is_recording=False)
>>>>>>> 394c588069822f5223619507530a11834afb0232
        print("reset")
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            # action1 = np.random.uniform(-1, 1)
            action2 = np.random.uniform(-1, 1)
            action1 = 0
            # action2 = 0
            state, reward, done, info = env.step([action1,action2])
            time.sleep(0.01)
    finally:
        env.close()

if __name__ == "__main__":
    main()