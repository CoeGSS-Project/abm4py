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

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

class Writer():
    
    def __init__(self, world, filename):
        self.fid = open('output/' + filename,'w')
        self.world = world
        self.world.reporter.append(self)
        
    def write(self,outStr):
        self.fid.write(str(self.world.time) + ' - ' + outStr + '\n')
        
    def close(self):
        self.fid.close()

class Memory():
    
    def __init__(self, memeLabels):
        self.freeRows  = list()
        self.columns   = dict()
        for i, label in enumerate(memeLabels):
            self.columns[label] = i
        self.ID2Row    = dict()
        self.memory    = np.zeros([0, len(memeLabels)])
        self.getID     = itertools.count().next

    def addMeme(self, meme):
        """
        add meme to memory
        """
        memeID = self.getID()
        if len(self.freeRows) > 0:
            row = self.freeRows.pop()
            self.memory[row] = meme
            self.ID2Row[memeID] = row
        else:
            row = self.memory.shape[0]
            self.memory = np.vstack(( self.memory, meme))
            self.ID2Row[memeID] = row
        
        return memeID
            
    def remMeme(self,memeID):
        """
        remove meme to memory
        """
        row = self.ID2Row[memeID]
        self.memory[row] = np.nan
        self.freeRows.append(row)
        del self.ID2Row[memeID]
    
    def getMeme(self,memeID,columns):
        """
        used the memeID to identfy the row and returns the meme
        """
        cols = [self.columns[x] for x in columns]
        rows = [self.ID2Row[x]   for x in memeID]
        return self.memory[np.ix_(rows,cols)]
    
class Record():
    
    def __init__(self, name, colLables, nSteps, title, style='plot'):
        self.nRec = len(colLables)
        self.columns = colLables
        self.rec = np.zeros([nSteps, self.nRec])
        self.name = name
        self.style = style
        self.nSteps = nSteps
        self.title  = title
    
    def set(self, time, data):
        self.rec[time,:] = data
        
    def setIdx(self, time, data, idx):
        self.rec[time,idx] = data
    
    def add(self, time, data):
        self.rec[time,:] += data
        
    def addIdx(self, time, data, idx):
        self.rec[time,idx] += data
    
    def div(self, time, data):
        self.rec[time,:] /= data
        
    def divIdx(self, time, data, idx):
        self.rec[time,idx] /= data
        
    def plot(self):
        plt.figure()
        if self.style == 'plot':
            plt.plot(self.rec)
        elif self.style == 'stackedBar':
            nCars = np.zeros(self.nSteps)
            colorPal =  sns.color_palette("Set3", n_colors=len(self.columns), desat=.8)
            for i, brand in enumerate(self.columns):
               plt.bar(np.arange(self.nSteps),self.rec[:,i],bottom=nCars, color =colorPal[i], width=1)
               nCars += self.rec[:,i]
            
        plt.legend(self.columns)
        plt.title(self.title)
        plt.savefig('output/' + self.name + '.png')
        
    def saveCSV(self):
        df = pd.DataFrame(self.rec, columns=self.columns)
        df.to_csv('output/' + self.name + '.csv')
        