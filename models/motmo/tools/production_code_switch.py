#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:29:55 2018

@author: geiges
This file switches between production code and development code
"""
import sys
if len(sys.argv) > 1:
    flag = sys.argv[1]
else:
    flag = 1


fileNames = list()

fileNames.append('../init_motmo.py')
fileNames.append('../classes_motmo.py')
fileNames.append('../../../lib/lib_gcfabm.py')

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
            line = line.replace('lib_gcfabm', 'lib_gcfabm_prod')
            line = line.replace('classes_motmo', 'classes_motmo_prod')
            if '##OPTPRODUCTION' in line:
                
                outLineList.append('#' + line)
                                 
            else:
                outLineList.append(line)
                
    fidOut.writelines(outLineList)
    fidOut.close()