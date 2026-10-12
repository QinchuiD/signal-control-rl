import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

render_mode = "human" if args.render else None
env = gym.make('CartPole-v1', render_mode="human")
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.fc1=nn.Linear(4,128)

        #Actor的全连接层
        self.action_head=nn.Linear(128,2)
        #价值函数的全连接层
        self.value_head=nn.Linear(128,1)

        self.save_actions=[]
        self.rewards=[]

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)

        action_logits=self.action_head(x)
        state_values=self.value_head(x)

        return action_logits,state_values

model=Policy()
optimizer=optim.Adam(model.parameters(),lr=3e-2)
eps=1e-10

def select_action(state):
    state=torch.from_numpy(state).float()
    logits,state_value=model(state)
    m=Categorical(logits=logits)
    action=m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action),state_value))
    return action.item()

def finish_episode():
    R=0
    saved_actions=model.save_actions
    policy_losses=[]
    value_losses=[]
    returns=[]

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns=torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob,value),R in zip(saved_actions,returns):
        advantage=R-value.item()

        policy_losses.append(-log_prob*advantage)
        value_losses.append(F.smooth_l1_loss(value,torch.tensor([R])))

        optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]
def main():
    running_reward=10

    for i_episode in count(1):

        state,_=env.reset()
        ep_reward=0

        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            ep_reward += (float)(reward)
            if terminated or truncated:
                break

        running_reward=0.05*ep_reward+(1-0.05)*running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
if __name__ == '__main__':
    main() 
