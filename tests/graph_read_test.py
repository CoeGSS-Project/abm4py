#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:35:31 2018

@author: gcf
"""

import igraph  as ig
import numpy as np
from  timeit import timeit as ti
#%%  Setup
graph = ig.Graph()

nVert = 1000
graph.add_vertices(nVert)

for vertice in graph.vs:
    vertice['property'] = np.random.random(5)
    #print vertice

#creation of a arbitray sequence
vertexSequence = graph.vs[np.arange(0,nVert,5).tolist()]

#%% reading the properties of a sequence to an numpy array


#first possiblity using asarray
%timeit numpyArray2D = np.asarray(vertexSequence['property'],dtype=np.float64)

#second possiblity with pre-allocated array and numpy view
numpyArray2D = np.zeros([len(vertexSequence),5])
%timeit numpyArray2D[:] = vertexSequence['property']

#%%