# Reward Shaping
## sparse shaping
在前面的情景中，都有一个前提，每一个action都能获得相应的奖励，但在特殊场景下，只有最后结束时才能获得奖励，其余时刻action所能获得的奖励都是0。比如下围棋，只有最后胜利才能获得reward。    
面对这种情况，就需要人为对这些步骤施加奖励

## reward-shaping Curiosity
对模型的好奇心赋予reward，每当模型观测到一些新东西，就施加reward

# No reward
对于一些真实情景下的任务，reward的定义是十分困难的，人为设定reward可能招致不可控的表现
## imitation learning
如果actor与environment互动无法获得reward，我们可以找一些expert也与环境互动，并记录下来每个episode，作为示范，让模型在与environment的互动中对示范进行学习     
但这不代表要模型完全模仿expert的行为，这会带来两个问题：第一，有一些极端情况会很少被记录下来，模型就学习不到极端情况的处理方式；第二，expert的行为不完全是对当前状态有效的，模型可能也会学习下来，更糟糕的是，模型学习能力有限且没有甄别能力，可能导致学习到无效行为但没学会有效的行为

## Inverse Reinforcement Learning
经典的rl问题，通常是已知环境，奖励函数，设定合适的强化学习算法，训练出最佳的行动，但在无reward的情景下，借助expert的行动记录，加上已有的环境，强化学习算法，可以反推出奖励函数
![alt text](image.png)