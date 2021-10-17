#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict
import torch
import dill


# In[3]:


class FirstVisitMC(object):
    '''
    On-Policy First-Visit MC Control
    '''
    def __init__(self,action_dim,cfg):
        self.action_dim=action_dim
        self.epsilon=cfg.epsilon
        self.gamma=cfg.gamma
        self.Q_table=defaultdict(lambda: np.zeros(action_dim))
        self.returns_sum=defaultdict(float) #对于回报进行求和
        self.returns_count=defaultdict(float)
        
    def choose_action(self,state):
        '''e-greedy policy'''
        if state in self.Q_table.keys():
            best_action=np.argmax(self.Q_table[state]) #取出的是一个索引
            action_probs=np.ones(self.action_dim,dtype=float)*self.epsilon/self.action_dim #计算每个动作的概率，就是一个均匀概率
            action_probs[best_action]+=(1.0-self.epsilon) #将贪心的概率赋值给最好的动作
            action=np.random.choice(np.arange(len(action_probs)),p=action_probs) #从动作概率中随机采样
        else:
            action=np.random.randint(0,self.action_dim)
        
        return action
    
    
    def update(self,one_ep_transition):
        '''
        以每个回合为周期进行转移
        将每个状态转换成元组，这样就可以通过字典进行一个取值
        '''
        sa_in_episode=set([(tuple(x[0]),x[1]) for x in one_ep_transition])
        
        for state, action in sa_in_episode:
            sa_pair=(state,action)
            #找到在每一个回合中第一个发生的对
            first_occur_idx=next(i for i,x in enumerate(one_ep_transition) if x[0]==state and x[1]==action)
            #将每一个回合的回报都相加
            G=sum([x[2]*(self.gamma**i) for i,x in enumerate(one_ep_transition[first_occur_idx:])])
            # 在所有采样的回合中计算该状态的平均回报
            self.returns_sum[sa_pair]+=G
            self.returns_count[sa_pair]+=1.0
            self.Q_table[state][action]=self.returns_sum[sa_pair]/self.returns_count[sa_pair]
            
    def save(self,path):
        '''
        把Q表格点的数据保存到文件中
        '''
        torch.save(
            obj=self.Q_table,
            f=path+"Q_table",
            pickle_module=dill
        )
    
    def load(self,path):
        self.Q_table=torch.load(
            f=path+"Q_table",
            pickle_module=dill
        )
        


# In[ ]:




