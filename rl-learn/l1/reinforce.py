import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# argparse 命令行参数解析模块

# 创建一个参数解析器，并添加对程序的描述
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example') 
# 添加gamma参数，设定默认值
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',   
                    help='discount factor (default: 0.99)')
# 设定随机种子
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
# 是否显示环境画面
parser.add_argument('--render', action='store_true',
                    help='render the environment')
# 设置日志间隔
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
# 读取命令行参数，并将其存储在args对象中
args = parser.parse_args() 

render_mode = "human" if args.render else None
# 创建环境
env = gym.make('CartPole-v1', render_mode="render_mode")  
env.reset(seed=args.seed)

# 设置统一种子，保证可复现
torch.manual_seed(args.seed)

# 定义策略网络
class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.fc1=nn.Linear(4,128)
        self.dp=nn.Dropout(p=0.6)
        self.fc2=nn.Linear(128,2)

        self.saved_log_probs=[]
        self.rewards=[]

    def forward(self,x):
        x=self.fc1(x)
        x=self.dp(x)
        x=F.relu(x)
        action_scores=self.fc2(x)

        return F.softmax(action_scores,dim=1)

policy=Policy()
optimizer=optim.Adam(policy.parameters(),lr=1e-2)
eps=1e-9

# action的选取
def select_action(state):
    # 把状态转换为张量，并添加一个维度，匹配网络的输入
    state=torch.from_numpy(state).float().unsqueeze(0)
    probs=policy(state)
    m=Categorical(probs)
    action=m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R=0
    returns=deque()
    for r in policy.rewards[::-1]:
        R=r+args.gamma*R
        returns.appendleft(R)

    returns=torch.tensor(returns)
    returns=(returns-returns.mean())/(returns.std()+eps)

    policy_loss=[]
    for log_prob,R in zip(policy.saved_log_probs,returns):
        policy_loss.append(-log_prob*R)

    optimizer.zero_grad()
    policy_loss=torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0.0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += (float)(reward)
            if terminated or truncated:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')



if __name__ == '__main__':
    main()