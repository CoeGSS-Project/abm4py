#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:37:19 2017

@author: gcf
"""

import multiprocessing as multi
from multiprocessing import Manager
import numpy as np 
manager = Manager()

glob_data= manager.list([])

def func(a):
    glob_data.append(a)

map(func,range(10))
print glob_data  #[0,1,2,3,4 ... , 9]  Good.

p=multi.Pool(processes=2)
p.map(func,range(80))

print glob_data




#%%

p=multi.Pool(processes=2)


def func(agent):
    return agent.weightFriendExperience(agent)

agList = earth.iterNodes(_pers)
p.map(func,agList) 

#%%
p=multi.Pool(processes=2)


def func(edge):
    return edge.nID(agent)

agList = earth.iterNodes(_pers)
p.map(func,edgesAll) 