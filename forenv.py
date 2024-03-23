import keyboard
from matplotlib.animation import FuncAnimation
from tensorboardX import SummaryWriter
from wifi_streaming import Env, client_order
import time
import numpy as np

def main():
    try:
        writer = SummaryWriter("runs/recording_EXO_newEMG_plot")
        env = Env.ExoskeletonEnv(writer, device='cuda')
        state = env.reset()
        print("reset")
        done = False
        while not done:
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
            # action1 = np.random.uniform(-1, 1)
            # action2 = np.random.uniform(-1, 1)
            action1 = 0
            action2 = 0
            state, reward, done, info = env.step([action1,action2])
            print("R_angle: ", state[0], "L_angle: ", state[3],"reward: ",reward)
            time.sleep(0.001)
    finally:
        client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
        print("disconnect")
        env.log_writer.close()

if __name__ == "__main__":
    main()