#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:24:30 2017

@author: gcf
"""

import h5py
import os
import numpy as np

resourcePath = 'resources_ger/'

regionIdRaster = np.load(resourcePath + 'subRegionRaster_186x219.npy') 
regionIDList = np.unique(regionIdRaster[~np.isnan(regionIdRaster)]).astype(int)  

for re in regionIDList:
    
    h5File = h5py.File(resourcePath + 'people' + str(re) + '.hdf5')
    dset = h5File.get('people')
    data = dset[:]
    
    print type(data[0,0])

    for col in range(data.shape[1]):
        if np.any(np.isnan(data[:,col])):
            idx = np.where(np.isnan(data[:,col]))[0]
            print 'nan found in file: ' + str(re) + ' in column: ' + str(col) + ' for indices: ' + str(idx)