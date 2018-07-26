#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

This file is part of GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://www.gnu.org/licenses/>.
"""

from numba import njit
import numpy as np
import pickle

class synthInput():
    """ MPI conform loading of synthetic population data"""
    pass # ToDo


    
    
class Record():
        """ 
        This calls manages the translation of different graph attributes to
        the output format as a numpy array. Vectora of values automatically get
        assigned the propper matrix dimensions and indices.

        So far, only integer and float are supported
        """
        def __init__(self, 
                     nAgents, 
                     agIds, 
                     nAgentsGlob, 
                     loc2GlobIdx, 
                     agTypeID, 
                     timeStepMag):
            
            self.ag2FileIdx = agIds
            self.nAgents = nAgents
            self.nAttr = 0
            self.attributeList = list()
            self.attrIdx = dict()
            self.header = list()
            self.timeStep = 0
            self.nAgentsGlob = nAgentsGlob
            self.loc2GlobIdx = loc2GlobIdx
            self.agTypeID    = agTypeID
            self.timeStepMag = timeStepMag


        def addAttr(self, name, nProp):
            attrIdx = list(range(self.nAttr,self.nAttr+nProp))
            self.attributeList.append(name)
            self.attrIdx[name] = attrIdx
            self.nAttr += len(attrIdx)
            self.header += [name] * nProp

        def initStorage(self, dtype):
            #print dtype
            self.data = np.zeros([self.nAgents, self.nAttr ], dtype=dtype)

        def addData(self, timeStep, nodeData):
            self.timeStep = timeStep
            self.data = nodeData[self.ag2FileIdx][self.attributeList]

        def writeData(self, h5File, folderName=None):
            #print self.header
            if folderName is None:
                path = '/' + str(self.agTypeID)+ '/' + str(self.timeStep).zfill(self.timeStepMag)
                #print 'IO-path: ' + path
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data
            else:
                path = '/' + str(self.agTypeID)+ '/' + folderName
                #print 'IO-path: ' + path
                self.dset = h5File.create_dataset(path, (self.nAgentsGlob,), dtype=self.data.dtype)
                self.dset[self.loc2GlobIdx[0]:self.loc2GlobIdx[1],] = self.data
                
                
@njit
def weightingFunc(x,y):
    return 1./((x**2. +y**2.)**.5)

@njit
def distance(x,y):
    return (x**2. +y**2.)**.5




def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out



def saveObj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)    