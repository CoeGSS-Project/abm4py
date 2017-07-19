#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:11:14 2017

@author: gcf
"""

import numpy as np

peerDict = dict()
peerDict[0] = list('412')
peerDict[1] = list('0235')
peerDict[2] = list('0135')
peerDict[3] = list('125')
peerDict[4] = list('0')
peerDict[5] = list('123')

nPeers = [len(peer) for peer in peerDict.values()]
#%%
sequence = np.argsort(nPeers)
sequence = sequence[-1:0:-1]

commSize= (len(peerDict), max(nPeers))

commArray = np.zeros(commSize)* np.nan
i=0
j=0

def setCommArray(commArray, peerDict, iStart,jStart):
    success = False
    if jStart+1 > len(peerDict[iStart]):
        iStart +=1
        jStart= -1
    for i in range(iStart,len(sequence)):
        me   = sequence[i]
        peers = peerDict[me]

        for j in range(jStart+1,len(peers)):
            peer = int(peers[j])
    
            if peer in commArray[me,:]:
                continue
    
    
            for col in range(commArray.shape[1]):
                if np.isnan(commArray[me,j]):
                    if me not in commArray[:,col] and peer not in commArray[:,col]:
                        #set communication
                        commArray[me,col] = peer
                        print peer, col
                        
                        commArray[peer,col] = me
                        success = True
                        break
            if success:
                commArray = setCommArray(commArray, peerDict, i, j)
                

    return commArray
            
commArray = setCommArray(commArray, peerDict, 0, -1)    
    