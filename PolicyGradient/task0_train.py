#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import gym
import torch
import datetime
from itertools import count
from agent import PolicyGradient
from plot import plot_rewards_cn
from utils import save_results,make_dir


# In[ ]:


curr_path=os.path.dirname(os.path.abspath(__file__))
curr_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# In[ ]:


class PGConfig(object):
    def __init__(self):
        self.algo="PolicyGradient" #项目名字
        self.env="CartPole-v0" #环境名称
        self.result_path=curr_path+"/outputs/"+self.env+"/"+curr_time+"/results/" #结果保存路径
        self.model_path=curr_path+"/outputs/"+self.env+"/"+curr_time+"/models/" #模型保存的路径
        self.train_eps=300 #训练的episode数目
        self.eval_eps=50
        self.batch_size=8
        self.lr=0.01 #学习率
        self.gamma=0.99
        self.hidden_dim=36
        self.device=torch.device("cuda")
        
        
def env_agent_config(cfg,seed):
    env=gym.make(cfg.env)
    env.seed(seed)
    state_dim=env.observation_space.shape[0]
    agent=PolicyGradient(state_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print("测试开始！")
    print(f"环境：{cfg.env},算法：{cfg.algo},设备：{cfg.device}")
    state_pool=[] #存放每batch_size个episode的state序列
    action_pool=[]#存放每batch_size个episode的action序列
    reward_pool=[]#存放每batch_size个episode的reward序列
    rewards=[]
    ma_rewards=[]
    for i_episode in range(cfg.train_eps):
        state=env.reset()
        ep_reward=0
        for _ in count():#作用等同于while True
            action=agent.choose_action(state) #根据当前环境state选择action
            next_state,reward,done,_=env.step(action)
            ep_reward+=reward
            if done:
                reward=0
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state=next_state
        
            if done:
                print("回合:",i_episode,"回报：",ep_reward)
                break
        print(len(action_pool))
        if i_episode>0 and i_episode % cfg.batch_size==0:
            agent.update(reward_pool,state_pool,action_pool)
            state_pool=[] #每个回合的state
            action_pool=[] #每个回合的action
            reward_pool=[] #每个回合的reward
        print(len(action_pool))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print("训练完毕！")
    return rewards,ma_rewards


def eval(cfg,env,agent):
    print("测试开始！")
    print(f"环境：{cfg.env},算法：{cfg.algo},设备：{cfg.device}")
    rewards=[]
    ma_rewards=[]
    for i_episode in range(cfg.eval_eps):
        state=env.reset()
        ep_reward=0
        for _ in count():
            env.render()
            action=agent.choose_action(state)
            next_state,reward,done,_=env.step(action)
            ep_reward+=reward
            if done:
                reward=0
            state=next_state
            if done:
                print("回合:",i_episode,"回报：",ep_reward)
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        env.close()
    print("测试完毕！")
    return rewards,ma_rewards        

        


# In[ ]:


cfg=PGConfig()

#训练
env,agent=env_agent_config(cfg,seed=10)
rewards,ma_rewards=train(cfg,env,agent)
make_dir(cfg.result_path,cfg.model_path)
agent.save(path=cfg.model_path)
save_results(rewards,ma_rewards,tag="train",path=cfg.result_path)
plot_rewards_cn(rewards,ma_rewards,tag="训练",algo=cfg.algo,path=cfg.result_path)

#测试
env,agent=env_agent_config(cfg,seed=10)
agent.load(path=cfg.model_path)
rewards,ma_rewards=eval(cfg,env,agent)
save_results(rewards,ma_rewards,tag="eval",path=cfg.result_path)
plot_rewards_cn(rewards,ma_rewards,tag="测试",env=cfg.env,algo=cfg.algo,path=cfg.result_path)


# In[ ]:




