from collections import deque
import numpy as np

class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class CustomExperienceSourceFirstLast:
    def __init__(self, env, agent, gamma=0.99, steps_count=1):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.buffer = []
        self.current_rewards = []
        self.current_states = []
        
    async def __aiter__(self):
        state = await self.env.async_reset()
        self.current_states = [state]
        self.current_rewards = [0.0]
        while True:
            action = await self.agent.select_action(state)
            next_state, reward, done, _ = await self.env.async_step(action)
            
            self.current_rewards[-1] += reward
            if len(self.current_states) == self.steps_count:
                exp = Experience(self.current_states[0], action, self.current_rewards[0],
                                 next_state, done)
                self.buffer.append(exp)
                self.current_rewards.pop(0)
                self.current_states.pop(0)
            self.current_rewards.append(0.0)
            self.current_states.append(next_state)
            
            if done:
                state = await self.env.async_reset()
                self.current_states = [state]
                self.current_rewards = [0.0]
            else:
                state = next_state
            
            while self.buffer:
                yield self.buffer.pop(0)

class CustomExperienceSourceFirstLast2:
    def __init__(self, env, agent, gamma=0.99, steps_count=1):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.buffer = []
        self.state = None
        self.action = None
        self.reward = 0.0
        self.done = False
        self.next_state = None

    async def update_buffer(self):
        self.state = await self.env.async_reset()
        step = 0
        total_reward = 0.0

        while True:
            self.action = await self.agent.select_action(self.state)
            self.next_state, reward, self.done, _ = await self.env.async_step(self.action)
            
            total_reward += (self.gamma ** step) * reward

            if self.done or step == self.steps_count - 1:
                self.buffer.append((self.state, self.action, total_reward, self.next_state, self.done))
                if self.done:
                    self.state = await self.env.async_reset()
                    total_reward = 0.0
                    step = 0
                else:
                    self.state = self.next_state
                    step += 1
            else:
                step += 1


    async def get_experience(self):
        if not self.buffer:
            await self.update_buffer()
        while self.buffer:
            yield self.buffer.pop(0)

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
