#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import torch
from collections import defaultdict


# In[2]:


class QLearning(object):
    
    def __init__(self,state_dim,action_dim,cfg):
        self.action_dim=action_dim #行动的维度
        self.lr=cfg.lr #学习率
        self.gamma=cfg.gamma
        self.epsilon=0
        self.sample_count=0
        self.epsilon_start=cfg.epsilon_start
        self.epsilon_end=cfg.epsilon_end
        self.epsilon_decay=cfg.epsilon_decay
        self.Q_table=defaultdict(lambda:np.zeros(action_dim)) #映射状态的嵌套字典
    
    def choose_action(self,state):
        self.sample_count+=1
        self.epsilon=self.epsilon_end+(self.epsilon_start-self.epsilon_end)*math.exp(-1.*self.sample_count/self.epsilon_decay)
        #epsilon是会递减的，这里选择指数递减
        # e-greedy策略
        if np.random.uniform(0,1)>self.epsilon:
            action=np.argmax(self.Q_table[str(state)]) #选择Q(s,a)最大对应的动作
        else:
            action=np.random.choice(self.action_dim) #随机选择动作
        return action
    
    def predict(self,state):
        action=np.argmax(self.Q_table[str(state)])
        return action
    
    def update(self,state,action,reward,next_state,done):
        Q_predict=self.Q_table[str(state)][action]
        if done:#终止状态
            Q_target=reward
        else:
            Q_target=reward+self.gamma*np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action]+=self.lr*(Q_target-Q_predict)
        
    def save(self,path):
        import dill
        torch.save(obj=self.Q_table,f=path+"Q_learning_model.pkl",pickle_module=dill)
        print("保存模型成功！")
        
    def load(self,path):
        import dill
        self.Q_table=torch.load(f=path+"Q_learning_model.pkl",pickle_module=dill)
        print("加载模型成功！")
        


# In[ ]:





# In[ ]:




