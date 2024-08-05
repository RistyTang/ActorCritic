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
    env = CacheResoureceSchdulingEnv()          # 初始状态

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
                        + str(env.all_configs[most_prob_action]) \
                        + ' most_prob_value ' + str(most_prob_value) +'\n\n'
            file_.write(log_info)
            if running_times == 200:
                log_info = '---------------change workloads here------------\n'
                file_.write(log_info)
        print('most_prob_action', most_prob_action)
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
    



    # '''
    # # 设置参数
    # learning_rate = 2e-2
    # num_episodes = 1000
    # hidden_dim = 16    # 隐藏层
    # gamma = 0.98    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    # "cpu")
    # state_dim = len(context)    # 表示每个状态的特征数目，此处设置18个
    # print("state_dim = ", state_dim)
    # action_dim = num_config   # 总的分配方式
    # print("action_dim = ", action_dim)

    # agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
    #               device)
    

    # for i in range(epochs):
    #     context, th_reward = get_now_reward(cache_hit)
    #     context_cache = np.array(context)

    #     # 转换状态为适应策略网络的输入格式
    #     state = torch.tensor(context_cache, dtype=torch.float).to(device)
    
    #     # 选择动作
    #     chosen_arm_cache = agent.take_action(state.cpu().numpy())  # 转换回 NumPy 数组以适应动作选择
    #     cache_index = chosen_arm_cache
    #     new_config = [curr_config[0], all_cache_config[cache_index]]

    #     print('----- new config:' + str(new_config[1]))
    #     cm.send_config(new_config)
    #     time.sleep(0.1)

    #     # 等待结果
    #     curr_config = cm.receive_config()
    #     app_name = curr_config[0]
    #     cache_config = curr_config[1]
    #     cache_hit = curr_config[2]
    #     cache_reward = curr_config[3]

    #     context, th_reward = get_now_reward(cache_hit)
    #     print('----- reward: ' + str(th_reward))
    #     context_cache = np.array(context)
    #     state = torch.tensor(context_cache, dtype=torch.float).to(device)
    
    #     # 更新策略
    #     agent.update({
    #         'rewards': [th_reward],
    #         'states': [state.cpu().numpy()],
    #         'actions': [chosen_arm_cache]
    #     })

    #     # 写入日志
    #     log_info = str(i) + ' ' + str(th_reward) + '\n'
    #     file_.write(log_info)

    #     if (i + 1) % 10 == 0:
    #         print('epoch [{}/{}]'.format(i + 1, epochs))

    # file_.close()
    # '''
    