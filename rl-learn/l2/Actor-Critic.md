# Critic
根据当前的Actor，Critic可以根据当前的状态或者状态+动作来预测出最终会获得的total reward
设计函数$V^\theta(s)$,输入s，输出reward的预测值
# 如何训练出一个 $V^\theta(s)$
## 蒙特卡洛方法MC
让Actor和Environment交互，获得真实的reward G，减小 $V^\theta(s)$ 预测值与真实值的误差
## 时序差分算法TD
TD不要求进行完整的一次交互，只需要知道当前状态$s_t$，action a，reward r，以及下一个状态$s_{t+1}$即可    
根据G的定义可以得到以下两个式子    
$$V^{\theta}(s_t)=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}...$$
$$V^{\theta}(s_{t+1})=r_{t+1}+\gamma r_{t+2}...$$
所以可得
$$V^\theta(s_t)-\gamma V^\theta(s_{t+1})=r_t$$
根据当前的$V^{\theta}$预测出这两个值并根据式子使差接近$r_t$

# $V^\theta(s)$怎么用来训练Actor
在Actor中对A的设计中，需要对A进行一次正则化，那么具体的操作就是对当前的G减去一个数，这个数就是$V^\theta(s)$，因为$V^\theta(s)$是对当前环境的预测，相当于是对不同action的平均值，大于平均值就认为是一个好的选择
但目前依然存在问题，因为此时的G是一个随机采样出的值，不能完全代表这个action的好坏，所以也希望用这个action获得的total reward的平均值来计算，所以根据上面式子可以得到G的平均值为$r_t+V^{\theta}(s_{t+1})$    
A的最终形态为：
$$A_t=r_t+V^\theta(s_t+1)-V^\theta(s_t)$$
