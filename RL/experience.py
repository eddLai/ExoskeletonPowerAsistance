import gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

import asyncio
Experience = collections.namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class AsyncExperienceSource:
    def __init__(self, env, agent, steps_count=2, steps_delta=1):
        self.env = [env] if not isinstance(env, list) else env
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []

    async def __aiter__(self):
        states, histories, cur_rewards, cur_steps = [], [], [], []
        for env in self.env:
            state = await env.async_reset()
            states.append(state)
            histories.append(deque(maxlen=self.steps_count))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        while True:
            actions = await self.agent(states)
            for idx, env in enumerate(self.env):
                next_state, reward, is_done, _ = await env.async_step(actions[idx])
                cur_rewards[idx] += reward
                cur_steps[idx] += 1

                history = histories[idx]
                history.append(Experience(state=states[idx], action=actions[idx], reward=reward, next_state=next_state, done=is_done))

                if is_done or len(history) == self.steps_count:
                    yield tuple(history)
                    histories[idx] = deque(maxlen=self.steps_count)
                    self.total_rewards.append(cur_rewards[idx])
                    self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                    states[idx] = await env.async_reset()
                else:
                    states[idx] = next_state

class AsyncExperienceReplayBuffer:
    def __init__(self, exp_source, buffer_size=10000):
        self.exp_source = exp_source
        self.buffer = []
        self.capacity = buffer_size

    async def populate(self, num_experiences=1):
        for _ in range(num_experiences):
            async for experience in self.exp_source.get_experience():
                self.buffer.append(experience)
                if len(self.buffer) > self.capacity:
                    self.buffer.pop(0)

    async def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
