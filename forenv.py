import keyboard
from matplotlib.animation import FuncAnimation
from tensorboardX import SummaryWriter
from wifi_streaming import Env, client_order

def main():
    try:
        writer = SummaryWriter("runs/testforenv2")
        env = Env.ExoskeletonEnv(writer, device='cuda')
        # state = env.reset()
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
    finally:
        client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
        env.log_writer.close()

if __name__ == "__main__":
    main()