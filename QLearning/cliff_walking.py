#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
from gym.envs.toy_text import discrete


# In[2]:


UP=0
RIGHT=1
DOWN=2
LEFT=3

class CliffWalkingEnv(discrete.DiscreteEnv):
    
    metadata={"render.modes":["human","ansi"]}
    
    def _limit_coordinates(self,coord):
        coord[0]=min(coord[0],self.shape[0]-1)
        coord[0]=max(coord[0],0)
        coord[1]=min(coord[1],self.shape[1]-1)
        coord[1]=max(coord[1],0)
        return coord
    
    def _calculate_transition_prob(self,currrent,delta):
        new_position=np.array(current)+np.array(delta)
        new_position=self._limit_coordinates(new_position).astype(int)
        new_state=np.ravel_multi_index(tuple(new_position),self.shape)
        reward=-100.0 if self._cliff[tuple(new_position)] else -1.0
        is_done=self._cliff[tuple(new_position)] or (tuple(new_position)==(3,11))
        return [(1.0,new_state,reward,is_done)]
    
    def __init__(self):
        self.shape=(4,12)
        
        nS=np.prod(self.shape) #计算乘积
        action_dim=4
        
        #Cliff Location
        self._cliff=np.zeros(self.shape,dtype=np.bool)
        self._cliff[3,1:-1]=True #设置悬崖路径
        
        #Calculate transition probabilities
        P={}
        for s in range(nS):
            position=np.unravel_index(s,self.shape) #返回位置索引，格式是坐标形式
            P[s]={a:[] for a in range(action_dim)}
            P[s][UP]=self._calculate_transition_prob(position,[-1,0])
            P[s][RIGHT]=self._calculate_transition_prob(position,[0,1])
            P[s][DOWN]=self._calculate_transition_prob(position,[1,0])
            P[s][LEFT]=self._calculate_transition_prob(position,[0,-1])
            
        #总是从状态(3,0)开始
        isd=np.zeros(nS)
        isd[np.ravel_multi_index((3,0),self.shape)]=1.0
        
        super(CliffWalkingEnv,self).__init__(nS,action_dim,P,isd)
        
    def render(self,mode="human",close=False):
        self._render(mode,close)
        
    def _render(self,mode="human",close=False):
        if close:
            return 
        outfile=StringIO() if mode=="ansi" else sys.stdout
        
        for s in range(self.nS):
            position=np.unravel_index(s,self.shape)
            
            if self.s==s:
                output=" x "
            elif position==(3,11):
                output=" T " 
            elif self._cliff[position]:
                output=" C "
            else:
                output=" o "
            
            if position[1]==0:
                output=output.lstrip()
            if position[1]==self.shape[1]-1:
                output=output.rstrip()
                output+="\n"
            
            outfile.write(output)
        outfile.write("\n")
                


# In[ ]:




