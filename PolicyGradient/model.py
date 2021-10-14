#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class MLP(nn.Module):
    '''
    多层感知机
    输入：state维度
    输出：概率
    '''
    def __init__(self,state_dim,hidden_dim=36):
        super(MLP,self).__init__()
        #24和36为隐藏层的层数，可根据state_dim,action_dim的情况来改变
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.sigmoid(self.fc3(x))
        return x


# In[ ]:




