#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from pathlib import Path


# In[2]:


def save_results(rewards,ma_rewards,tag="train",path="./results"):
    '''
    将回报和平均回报保存下来
    '''
    np.save(path+"{}_rewards.npy".format(tag),rewards)
    np.save(path+"{}_ma_rewards.npy".format(tag),ma_rewards)
    print("结果保存完毕！")
    

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True,exist_ok=True)
    
def del_empty_dir(*paths):
    for path in paths:
        dirs=os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path,dir)):
                os.removedirs(os.path.join(path,dir))


# In[ ]:




