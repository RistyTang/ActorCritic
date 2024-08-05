import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# 创建一个策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# 策略梯度算法
class PolicyGradientAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.policy = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        action = np.random.choice(self.env.action_space.n, p=probs.detach().numpy()[0])
        return action, probs[0, action]

    def update_policy(self, rewards, log_probs):
        discounted_rewards = self.compute_discounted_rewards(rewards)
        loss = -torch.stack(log_probs).sum() * torch.tensor(discounted_rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        running_sum = 0
        for r in reversed(rewards):
            running_sum = r + self.gamma * running_sum
            discounted_rewards.insert(0, running_sum)
        return discounted_rewards

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            rewards = []
            log_probs = []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state

            self.update_policy(rewards, log_probs)
            print(f'Episode {episode + 1}/{num_episodes} finished')

if __name__ == "__main__":
    agent = PolicyGradientAgent('CartPole-v1')
    agent.train(num_episodes=1000)
