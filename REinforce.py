import rl_utils
import torch
import torch.nn.functional as F
import numpy as np
import itertools
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
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()        # 梯度置零
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
    def __init__(self, app_num, resource_num, workload_change, step_num = 1):
        '''
        app_num : 任务数量
        resource_num : 总资源单位数目
        workload_change : 工作负载开始变化轮次
        step_num : 单次调整步长
        '''
        self.app_num = app_num
        self.num_resources = resource_num                # 总资源数目
        self.counter = 0
        self.curr_count = 0                     # 当前epoch运行次数
        self.workload_change = workload_change
        self.allconfigs = []
        self.step_num = step_num
        for i, j in itertools.permutations(range(app_num), 2):
            mod = [0] * app_num
            mod[i] += step_num
            mod[j] -= step_num
            self.allconfigs.append(mod)

        self.allconfigs.append([0] * app_num)
        # print("all config", self.allconfigs)
        self.reset(0)

        
    def reset(self,curr_epoch):
        '''设置从均分开始调节，传入当前轮次实现负载变化'''
        self.state = []
        per_source_num = int(self.num_resources // self.app_num)
        for i in range(self.app_num):
            username = 'user' + str(i)
            self.state.append([username, per_source_num])
        cache_hit = []
        if curr_epoch < self.workload_change:
            cache_hit.append(userA_func(per_source_num, 50))
            cache_hit.append(userB_func(per_source_num, 100))
            cache_hit.append(userC_func(per_source_num, 100))
            cache_hit.append(userC_func(per_source_num, 150))   # 初始奖励6.5625
        
        else:
            cache_hit.append(userA_func(per_source_num, 150))
            cache_hit.append(userB_func(per_source_num, 100))
            cache_hit.append(userC_func(per_source_num, 100))
            cache_hit.append(userC_func(per_source_num, 50))   # 初始奖励6.3125
        
        # print(cache_hit)
        _, self.last_reward = get_now_reward(cache_hit)         # 上一步的奖励
        # print('epoch ', curr_epoch, 'max reward init', self.last_reward)
        return self.state
        
    def step(self, new_actions, curr_epoch):
        '''
        new_actions: 新传入的选择动作的下标
        curr_epoch: 当前与环境交互的回合
        '''
        
        self.curr_count = self.curr_count + 1
        done = False
            
        # TODO : 修改当前状态
        for i in range(self.app_num):
            self.state[i][1] += self.allconfigs[new_actions][i]
            # print(self.state[i])
        cache_hit = []
        if curr_epoch < self.workload_change:
            cache_hit.append(userA_func(self.state[0][1], 50))
            cache_hit.append(userB_func(self.state[1][1], 100))
            cache_hit.append(userC_func(self.state[2][1], 100))
            cache_hit.append(userC_func(self.state[3][1], 150))   # 初始奖励6.5625
        else:
            if curr_epoch == self.workload_change:
                print('-----------------------reward changes-----------------')
            cache_hit.append(userA_func(self.state[0][1], 150))
            cache_hit.append(userB_func(self.state[1][1], 100))
            cache_hit.append(userC_func(self.state[2][1], 100))
            cache_hit.append(userC_func(self.state[3][1], 50))   

        context, reward = get_now_reward(cache_hit)
        # 回报奖励是与最大值的差异 ->不稳定
        # 如果比上次回报值更大则为1，相反为-1
        if reward > self.last_reward:
            return_reward = 1
        else:
            # if reward == self.last_reward:
            #     return_reward = 0
            # else:
            return_reward = -1

        # 设置训练停止条件 : 1.某个任务没分配到资源
        for i in range(self.app_num):
            if self.state[i][1] <= self.step_num:
                done = True
                self.curr_count = 0
        # 2.计算轮次过多
        if  self.curr_count > 40 :
            # print('---------------while done-----run : ',self.curr_count)
            self.curr_count = 0
            done = True

        # if reward > self.last_reward:
        self.last_reward = reward

        return self.state, return_reward, done, context
    
    def GetDims(self):
        '''
        返回当前环境下的agent  state_dim & action_dim
        state_dim: 当前状态下每个任务已经分配的资源量 
        action_dim: 动作总数
        '''
        action_dim = self.app_num * (self.app_num - 1) + 1
        state_dim = self.app_num
        return state_dim, action_dim
    
    def PrintCurrState(self):
        print(self.state[0], self.state[1])
        print('reward', self.GetCurrReward())

    def GetCurrReward(self, curr_epoch):
        '''
        返回当前状态下的Reward
        '''
        cache_hit = []
        if curr_epoch < self.workload_change:
            cache_hit.append(userA_func(self.state[0][1], 50))
            cache_hit.append(userB_func(self.state[1][1], 100))
            cache_hit.append(userC_func(self.state[2][1], 100))
            cache_hit.append(userC_func(self.state[3][1], 150))   # 初始奖励6.5625
        else:
            cache_hit.append(userA_func(self.state[0][1], 150))
            cache_hit.append(userB_func(self.state[1][1], 100))
            cache_hit.append(userC_func(self.state[2][1], 100))
            cache_hit.append(userC_func(self.state[3][1], 50))   # 初始奖励6.5625
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
    
def userC_func(resources, threshold):
    if threshold < 0:
        return 0
    if resources < threshold:
        return resources * 0.080
    else:
        return threshold * 0.080 + (resources - threshold) * 0.015