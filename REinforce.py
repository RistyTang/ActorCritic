import rl_utils
import torch
import torch.nn.functional as F
import numpy as np
# from util import *

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 2,16
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 16,2
        # 初始化 fc2 的权重和偏置使得所有动作的初始概率相同
        torch.nn.init.constant_(self.fc2.weight, 0.0)  # 将权重初始化为零
        torch.nn.init.constant_(self.fc2.bias, 0.0)    # 将偏置初始化为零

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device, epsilon_start = 0.5, epsilon_end = 0.05, epsilon_decay=0.995):
        '''
        每个状态的特征数量，每个状态的特征数量，所有可能动作的数量
        模型参数调整的步长，平衡未来奖励和当前奖励的相对重要性
        '''
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)  # 前向网络
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        if np.random.rand() < self.epsilon:  # 探索
            action = np.random.randint(self.policy_net.fc2.out_features)
        else :
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

    # Convert lists to tensors
        reward_tensor = torch.tensor(reward_list, dtype=torch.float, device=self.device)
    
    # Calculate the cumulative return G
        G = 0
        returns = []
        for reward in reversed(reward_list):
            G = self.gamma * G + reward
            returns.insert(0, G)
    
    # Convert returns to tensor
        return_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)

    # Calculate advantage
        mean_return = return_tensor.mean()
        advantage_tensor = return_tensor - mean_return

        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # From last step
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            probs = self.policy_net(state)
            log_prob = torch.log(probs.gather(1, action))
        
        # Use advantage for loss calculation
            advantage = advantage_tensor[i]
            loss = -log_prob * advantage

            loss.backward()  # Backward pass to compute gradients
    
        self.optimizer.step()  # Update policy network parameters

    # Update epsilon if using epsilon-greedy strategy
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state_values = [value for _, value in state]
        state_tensor = torch.tensor([state_values], dtype=torch.float).to(self.device)
        probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # for actions, prob in enumerate(action_dist.probs[0]):
        #     print(f"动作 {actions}: 概率 {prob.item():.4f}")
        return action.item()

    def update(self, transition_dict):
        # print(transition_dict)
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    
    def GetMostProbAction(self, state):
        state_values = [value for _, value in state]
        state_tensor = torch.tensor([state_values], dtype=torch.float).to(self.device)
        probs = self.actor(state_tensor)
        most_prob_action = torch.argmax(probs, dim=1).item()
        most_prob_value = probs[0, most_prob_action].item()
        return most_prob_action, most_prob_value


class CacheResoureceSchdulingEnv:
    def __init__(self):
        self.app_num = 2
        self.num_resources = 32                # 总资源数目
        self.counter = 0
        self.all_configs = gen_all_config(self.app_num, self.num_resources)
        cache_hit = [userA_func(16, 8), userB_func(16, 24)]
        self.curr_count = 0                     # 当前epoch运行次数
        _, self.max_reward = get_now_reward(cache_hit)      # 初始奖励0.74
        self.reset(0)

        
    def reset(self,curr_epoch):
        '''回到初始状态，设置其为各自一半'''
        self.state = []
        self.state.append(['userA', self.num_resources // self.app_num])
        self.state.append(['userB', self.num_resources // self.app_num])
        if curr_epoch < 201:
            cache_hit = [userA_func(16, 8), userB_func(16, 24)]     # 初始奖励0.74
        else:
            cache_hit = [userA_func(16, 24), userB_func(16, 8)]     # 初始奖励0.94
        _, self.max_reward = get_now_reward(cache_hit)      
        # print('epoch ', curr_epoch, 'max reward init', self.max_reward)
        return self.state
        
    def step(self, new_actions, curr_epoch):
        '''
        new_actions: 新传入的选择动作的下标
        curr_epoch: 当前与环境交互的回合
        '''
        
        self.curr_count = self.curr_count + 1
        done = False
            
        # 修改当前状态
        # print('new_actions',self.all_configs[new_actions])
        for i in range(self.app_num):
            self.state[i][1] = self.all_configs[new_actions][i]
            # print(self.state[i])
        
        if curr_epoch < 201:
            cache_hit = [userA_func(self.all_configs[new_actions][0], 8), userB_func(self.all_configs[new_actions][1], 24)]
        else:
            if curr_epoch == 201:
                print('-----------------------reward changes-----------------')
            cache_hit = [userA_func(self.all_configs[new_actions][0], 24), userB_func(self.all_configs[new_actions][1], 8)]
        context, reward = get_now_reward(cache_hit)
        #  - self.max_reward
        return_reward = reward - self.max_reward

        # 设置训练停止条件是
        if (reward >= self.max_reward * 0.95 and self.curr_count >= 30) or self.curr_count > 50:
            # print('---------------while done-----run : ',self.curr_count)
            self.curr_count = 0
            done = True

        if reward > self.max_reward:
            self.max_reward = reward

        return self.state, return_reward, done, context
    
    def GetDims(self):
        '''
        返回当前环境下的agent  state_dim & action_dim
        state_dim: 当前状态下每个任务已经分配的资源量 
        action_dim: 动作总数
        '''
        return self.app_num, len(self.all_configs)
    
    def PrintCurrState(self):
        print(self.state[0], self.state[1])
        print('reward', self.GetCurrReward())

    def GetCurrReward(self, curr_epoch):
        '''
        返回当前状态下的Reward
        '''
        if curr_epoch < 201:
            cache_hit = [userA_func(self.state[0][1], 8), userB_func(self.state[1][1], 24)]
        else:
            cache_hit = [userA_func(self.state[0][1], 24), userB_func(self.state[1][1], 8)]
        _, reward = get_now_reward(cache_hit)
        return reward


def gen_all_config(num_apps, num_resources):
    '''
    generate all resource config according to the number of apps and total resources

    Args:
        num_apps (int): number of apps
        num_resources (int): total units of resources

    Returns:
        list<list>: a list containing all possible config, which is list<int>
    '''
    if num_apps == 1:
        # Only one app, it get all remaining resources
        return [[num_resources]]
    
    all_config = []
    for i in range(num_resources + 1):
        # Recursively allocate the remaining resources among the remaining app
        for sub_allocation in gen_all_config(num_apps - 1, num_resources - i):
            all_config.append([i] + sub_allocation)
    return all_config

def get_now_reward(curr_metrics):
    '''
    Get a default context and average reward

    Args:
        curr_metrics (list<float>): a feedback metric representing the current mixed deployment status for each app

    Returns:
        list<float>: context infomation, initialize as a list of 18 elements, each set to 1.0
        float: th_reward, the average reward calculated based on the current metrics
    '''
    context = [1.0 for _ in range(18)]      
    th_reward = sum(float(x) for x in curr_metrics)/len(curr_metrics)
    return context, th_reward


def userA_func(resources, threshold):
    if threshold < 0:
        return 0
    if resources < threshold:
        return resources * 0.095
    else:
        return threshold * 0.095 + (resources - threshold) * 0.010

def userB_func(resources, threshold):
    if threshold < 0:
        return 0
    if resources < threshold:
        return resources * 0.040
    else:
        return threshold * 0.040 + (resources - threshold) * 0.005