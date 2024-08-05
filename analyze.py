
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    """计算移动平均"""
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

if __name__ == '__main__':
    log_name = 'linUCB.log'
    index = []
    reward = []
    
    with open(log_name, 'r') as log_file:
        for line in log_file.readlines():
            parts = line.split()
            if len(parts) == 2: 
                idx, rew = line.split(' ')
                index.append(float(idx))
                reward.append(float(rew))
            

    # 转换为numpy数组
    index = np.array(index)
    reward = np.array(reward)
    
    # 设置移动平均窗口大小
    window_size = 10
    smoothed_reward = moving_average(reward, window_size)

    # 调整索引以匹配平滑后的数据长度
    smoothed_index = index[window_size - 1:]

    plt.figure()
    plt.plot(index, reward, linestyle='-', color='b', label='Original Data')
    plt.plot(smoothed_index, smoothed_reward, linestyle='-', color='r', label='Smoothed Data')
    plt.title('Actor-Critic schedule')
    plt.xlabel('epoch')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('reward_change.png')
    plt.show()