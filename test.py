import csv
import os
import sys
import time

import numpy as np
from REinforce import *

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



if __name__ == '__main__':
    # 环境设置
    env = CacheResoureceSchdulingEnv(4, 400, 201, 2)          # 初始状态
    print("before state", env.state)
    env.step(0, 1)
    print("after state", env.state)

    # agent设置
    state_dim, action_dim = env.GetDims()
    print('state_dim',state_dim,'action_dim',action_dim)
    hidden_dim = 16    
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.96
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)
    
    # 训练设置
    num_epochs = 40    # 每轮迭代次数
    file_ = open('linUCB.log', 'w', newline='')
    running_times = 0
    for i in range(10):                                  # 训练轮数
        for i_episode in range(num_epochs):      # 每一轮训练和环境交互的次数
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset(running_times)
            done = False
            running_times = running_times + 1
            while not done:
                action = agent.take_action(state)
                state_values = [value for _, value in state]
                next_state, reward, done, next_context = env.step(action, running_times)
                next_state_values = [value for _, value in next_state]
                transition_dict['states'].append(state_values)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state_values)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                # print('action : ',action,'state : ',state,'reward :',reward)
                state = next_state
            agent.update(transition_dict)
            log_info = str(running_times) + ' ' + str(env.GetCurrReward(running_times)) + '\n'
            file_.write(log_info)
            log_info = str(running_times) + ' curr_state is ' + str(env.state) +'\n'
            file_.write(log_info)
            most_prob_action, most_prob_value = agent.GetMostProbAction(env.state)
            log_info = str(running_times) + ' most_prob_action ' \
                        + str(env.allconfigs[most_prob_action]) \
                        + ' most_prob_value ' + str(most_prob_value) +'\n\n'
            file_.write(log_info)
            if running_times == 200:
                log_info = '---------------change workloads here------------\n'
                file_.write(log_info)
        print('current state is ', env.state)
        print('current reward is ', env.GetCurrReward(running_times))
        # log_info = str(running_times) + ' ' + str(env.GetCurrReward()) + '\n'
        # file_.write(log_info)
        # log_info = str(running_times) + ' curr_state is ' + str(env.state) +'\n'
        # file_.write(log_info)
        # most_prob_action, most_prob_value = agent.GetMostProbAction(env.state)
        # log_info = str(running_times) + ' most_prob_action ' \
        #             + str(env.all_configs[most_prob_action]) \
        #             + ' most_prob_value ' + str(most_prob_value) +'\n\n'
        # file_.write(log_info)
        # print('most_prob_action', most_prob_action)
        
    file_.close()
    