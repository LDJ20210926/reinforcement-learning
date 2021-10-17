#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os,sys
import torch
import datetime
from utils import save_results,make_dir
from plot import plot_rewards_cn
from agent import FirstVisitMC
from racetrack_env import RacetrackEnv


# In[ ]:


curr_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #获取当前时间
curr_path=os.path.dirname(os.path.abspath(__file__)) #当前路径

class MCConfig(object):
    def __init__(self):
        self.algo="MC"
        self.env = 'Racetrack'
        self.result_path = curr_path+"/outputs/" + self.env + '/'+curr_time+'/results/'  # 结果存储路径
        self.model_path = curr_path+"/outputs/" + self.env + '/'+curr_time+'/models/'  # 模型存储路径
        # 随机选择动作的概率:epsilon
        self.epsilon = 0.15
        self.gamma = 0.9  # gamma: 折现因子
        self.train_eps = 200
        self.device = torch.device("cuda" ) 

def env_agent_config(cfg,seed=1):
    env=RacetrackEnv()
    action_dim=9
    agent=FirstVisitMC(action_dim,cfg)
    return env,agent

def train(cfg,env,agent):
    print("开始训练！")
    print(f"环境：{cfg.env},算法：{cfg.algo},设备：{cfg.device}")
    rewards=[]
    ma_rewards=[]
    
    for i_ep in range(cfg.train_eps):
        state=env.reset()
        ep_reward=0
        one_ep_transition=[]
        while True:
            action=agent.choose_action(state)
            next_state,reward,done=env.step(action)
            ep_reward+=reward
            one_ep_transition.append((state,action,reward))
            state=next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        
        agent.update(one_ep_transition)
        if (i_ep+1) % 10==0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}:Reward:{ep_reward}")
    
    print("训练完毕!")
    return rewards,ma_rewards

def eval(cfg,env,agent):
    print("开始测试！")
    print(f"环境：{cfg.env},算法：{cfg.algo},设备：{cfg.device}")    
    rewards=[]
    ma_rewards=[]

    for i_ep in range(cfg.train_eps):
        state=env.reset()
        ep_reward=0
        while True:
            #env.render()
            action=agent.choose_action(state)
            next_state,reward,done=env.step(action)
            ep_reward+=reward
            state=next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        
        #env.close()
        if (i_ep+1) % 10==0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}:Reward:{ep_reward}")
    
    print("测试完毕!")
    return rewards,ma_rewards


# In[ ]:


cfg=MCConfig()

#训练
env,agent=env_agent_config(cfg,seed=1)
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




