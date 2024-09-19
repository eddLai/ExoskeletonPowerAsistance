import ptan
from data_streaming import Env
import torch
import time

class DummyAgent(ptan.agent.BaseAgent):
    """
    一個簡單的代理，總是返回零動作。
    """
    def __init__(self, action_dims):
        self.action_dims = action_dims

    def __call__(self, states, agent_states):
        return [torch.zeros(self.action_dims)], agent_states

class DummyNet(torch.nn.Module):
    """
    一個簡單的網絡，用於模擬TargetNet的行為。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(1)

if __name__ == "__main__":
    env = Env.ExoskeletonEnv("CUDA", "runs/ptantest")
    writer = env.log_writer
    agent = DummyAgent(action_dims=2)
    dummy_net = DummyNet()
    target_net = ptan.agent.TargetNet(dummy_net)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for idx, exp in enumerate(exp_source):
                time.sleep(0.05)

                if idx % 10 == 0:
                    target_net.sync()

                if exp.last_state is None:
                    tracker.reward(exp.reward, idx)
                    tb_tracker.track("reward", exp.reward, idx)
                
                if idx > 50:
                    break

    writer.close()
    print("Done")
