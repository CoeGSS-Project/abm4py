#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:22:32 2017

processing records
@author: geiges, GCF
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% 
path = 'output/'
files = os.listdir(path)

for filename in files:
    if filename.endswith('.csv'):
        df = pd.read_csv(path + filename)
        plt.figure()
        plt.plot(df)
        plt.title(filename)
