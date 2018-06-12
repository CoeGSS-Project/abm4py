#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:29:55 2018

@author: geiges
This file switches between production code and development code
"""
import sys
if len(sys.argv) > 1:
    flagOpt = bool(int(sys.argv[1]))
else:
    flagOpt = bool(1)

useNumba = False

fileNames = list()

fileNames.append('init_motmo.py')
fileNames.append('classes_motmo.py')
fileNames.append('../../lib/lib_gcfabm.py')
fileNames.append('../../lib/core.py')
fileNames.append('run-direct.py')
fileNames.append('scenarios.py')
fileNames.append('run-via-dakota.py')
fileNames.append('../../tests/scaling_test.py')

for fileName in fileNames:
    outLineList = []
    fidOut = open(fileName[:-3] + '_prod.py','w')
    with open(fileName) as fp:  
        line = fp.readline()
        cnt = 1     
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt += 1
            for prefix in ['import', 'from']:
                line = line.replace(prefix +' lib_gcfabm ', prefix +' lib_gcfabm_prod ')
                line = line.replace(prefix +' init_motmo ', prefix +' init_motmo_prod ')
                line = line.replace(prefix +' classes_motmo ', prefix +' classes_motmo_prod ')
                line = line.replace(prefix +' core ', prefix +' core_prod ')
                line = line.replace(prefix +' scenarios ', prefix +' scenarios_prod ')
            
            line = line.replace('import core', 'import core_prod as core')           
            line = line.replace('import scenarios', 'import scenarios_prod as scenarios') 
            if not useNumba:
                line = line.replace('@jit', '#ljit')
                line = line.replace('@njit', '#njit')
                line = line.replace('from numba', '#from numba')
                         
            if flagOpt and '##OPTPRODUCTION' in line:
                
                outLineList.append('#' + line)
                                 
            else:
                outLineList.append(line)
                
    fidOut.writelines(outLineList)
    fidOut.close()