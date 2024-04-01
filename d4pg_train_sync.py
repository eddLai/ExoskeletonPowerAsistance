import os
import ptan
import time
from wifi_streaming import Env
from RL import models
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import threading
from pynput import keyboard
from wifi_streaming import client_order

import torch
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10
REWARD_STEPS = 5 # 3~10

OBSERVATION_DIMS = 9+8
ACTION_DIMS = 2

TEST_ITERS = 160 # determines when training stop for a while
MAX_STEPS_FOR_TEST = 10

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

def find_best_model(base_path, subdir):
    """
    Searches for the best model within a specified directory.

    Parameters:
        base_path (str): The base path where models are stored.
        subdir (str): The subdirectory to search for the best model.

    Returns:
        tuple: Contains the path of the best model and its corresponding reward. 
               Returns (None, float('-inf')) if no model is found.
    """
    best_reward = float('-inf')  # Initialize the best reward to negative infinity
    best_model_path = None  # Initialize the best model path to None
    search_path = os.path.join(base_path, subdir)  # Full path to search in
    
    for file in os.listdir(search_path):  # Iterate through each file in the directory
        if file.startswith("best_") and file.endswith(".dat"):  # Check if file name matches the pattern
            try:
                reward_str = file.split('_')[1]  # Extract the reward value from the file name
                reward = float(reward_str)  # Convert the reward string to float
                if reward > best_reward:  # Update best reward and model path if a better reward is found
                    best_reward = reward
                    best_model_path = os.path.join(search_path, file)
            except ValueError:
                pass  # Ignore files where the reward value cannot be converted to float

    return best_model_path, best_reward  # Return the best model path and its reward

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    obs = env.reset(is_recording=False)
    # while True:
    #         obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
    #         mu_v = net(obs_v)
    #         action = mu_v.squeeze(dim=0).data.cpu().numpy()
    #         action = np.clip(action, -1, 1)
    #         obs, reward, done, _ = env.step(action)
    #         rewards += reward
    #         steps += 1
    #         if done or steps >= MAX_STEPS_FOR_TEST:
    #             print("net test1 finished")
    #             client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
    #             break
    # time.sleep(1)
    for i in range(count-1):
        steps = 0
        rewards = 0.0
        # obs = env.reset(is_recording=False)
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or steps >= MAX_STEPS_FOR_TEST:
                client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
                print(f"net test{i+2} finished")
                break
        time.sleep(1)
    return rewards / count, steps / count


def distr_projection(next_distr_v, rewards_v, dones_mask_t,
                     gamma, device="cpu"):
    # since we can't really computing tensor on cuda with numpy
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool_)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += \
            next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)

stop_event = threading.Event()
def on_press(key):
    try:
        if key.char == 'q':
            stop_event.set()
    except AttributeError:
        pass
def start_listening():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    start_listening()
    save_path = os.path.join("saves", "d4pg-" + args.name)
    actor_subdir = "actor"
    critic_subdir = "critic"
    os.makedirs(os.path.join(save_path, actor_subdir), exist_ok=True)
    os.makedirs(os.path.join(save_path, critic_subdir), exist_ok=True)

    act_net = models.DDPGActor(OBSERVATION_DIMS, ACTION_DIMS).to(device)
    crt_net = models.D4PGCritic(OBSERVATION_DIMS, ACTION_DIMS, N_ATOMS, Vmin, Vmax).to(device)

    best_actor_model_path, best_actor_reward = find_best_model(save_path, actor_subdir)
    best_critic_model_path, best_critic_reward = find_best_model(save_path, critic_subdir)

    if best_actor_model_path:
        print(f"best actor：{best_actor_model_path}, reward：{best_actor_reward}")
    else:
        print("No actor NN")

    if best_critic_model_path:
        print(f"best critic：{best_critic_model_path},reward：{best_critic_reward}")
    else:
        print("No critic NN")

    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg_" + args.name)
    env = Env.ExoskeletonEnv(log_writer=writer)
    agent = models.AgentD4PG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.SGD(act_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    training_stopped_early = False
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                if stop_event.is_set():
                    print("Training stopped by user.")
                    training_stopped_early = True
                    if best_reward is not None:
                        current_model_name = "best_%+.3f_%d.dat" % (best_reward, frame_idx)
                    else:
                        print("you stopped training before any best reward was achieved.")
                    actor_model_path = os.path.join(save_path, "actor", name)
                    critic_model_path = os.path.join(save_path, "critic", name)
                    torch.save(act_net.state_dict(), actor_model_path)
                    torch.save(crt_net.state_dict(), critic_model_path)
                    break

                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue
                if len(buffer) == REPLAY_INITIAL:
                    print("Initialization of the buffer is finished, start training...")
                    client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
                    input("Press Enter to continue...")

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, \
                dones_mask, last_states_v = \
                    models.unpack_batch(batch, device)

                # train critic
                crt_opt.zero_grad()
                crt_distr_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(
                    last_states_v)
                last_distr_v = F.softmax(
                    tgt_crt_net.target_model(
                        last_states_v, last_act_v), dim=1)
                proj_distr_v = distr_projection(
                    last_distr_v, rewards_v, dones_mask,
                    gamma=GAMMA**REWARD_STEPS, device=device)
                prob_dist_v = -F.log_softmax(
                    crt_distr_v, dim=1) * proj_distr_v
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v,
                                frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    client_order.FREEX_CMD(env.sock, "E", "0", "E", "0")
                    print("Please prepare for a test phase by changing the exoskeleton user, if desired.")
                    # input("Press Enter to continue after the user has been changed and is ready...")
                    ts = time.time()
                    rewards, steps = test_net(act_net, env, count=4, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            actor_model_path = os.path.join(save_path, "actor", name)
                            critic_model_path = os.path.join(save_path, "critic", name)
                            torch.save(act_net.state_dict(), actor_model_path)
                            torch.save(crt_net.state_dict(), critic_model_path)
                        best_reward = rewards
                time.sleep(0.01)
    # except KeyboardInterrupt:
        # print("Training interrupted by keyboard.")
    
    # finally:
    if best_reward is None:
        print("No best reward achieved during the training.")
    elif training_stopped_early:
        print(f"Training stopped, Best reward achieved: {best_reward:.3f}")
    try:
        env.close()
    except Exception as e:
        print(f"Error while closing resources: {e}")