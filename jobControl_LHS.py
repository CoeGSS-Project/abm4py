#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:40:54 2017

@author: gcf
"""

import pandas as pd
import pyDOE  as doe
from os.path import expanduser
home = expanduser("~")
from class_auxiliary import convertStr
paraDf = pd.read_csv('parameters_NBH.csv',index_col=0)


propListInt =  'urbanThreshold', 'urbanCritical'
propListFloat = 'individualPrio', 'initPriceBrown', 'convC', 'initialGreen','kappa'


nProp = len(propListInt) + len(propListFloat)
nReali = 200
maxDeviation = .20

factors = (doe.doe_lhs.lhs(nProp,nReali)*2 -1 ) * maxDeviation +1


for iReali in range(nReali): # loop over realizatons

    paraDfNew = paraDf.copy() # new copy of parameters
    iProp = 0
    for prop in propListInt: # loop over properties
        
        initValue = convertStr(paraDf.ix[prop]['value'])
        assert type(initValue) == int
        
        newValue = str(int(initValue * factors[iReali,iProp]))
        paraDfNew.ix[prop]['value'] = newValue
        iProp +=1
    
    for prop in propListFloat: # loop over properties
        
        initValue = float(convertStr(paraDf.ix[prop]['value']))
        
        
        newValue = str(initValue * factors[iReali,iProp])
        paraDfNew.ix[prop]['value'] = newValue
        iProp +=1

    paraDfNew.to_csv(home + '/python/agModel/doe/' + str(iReali) + 'input_para.csv')

import os
# job submission
for iReali in range(nReali)   :
    print 'sbatch ~/mobStart.sh ' + 'doe/' + str(iReali) + 'input_para.csv'
    os.system('sbatch ~/mobStart.sh ' + 'doe/' + str(iReali) + 'input_para.csv')