#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:11:14 2017

@author: gcf
"""

import numpy as np

peerDict = dict()
peerDict[1] = list('523')
peerDict[2] = list('1346')
peerDict[3] = list('1246')
peerDict[4] = list('236')
peerDict[5] = list('1')
peerDict[6] = list('234')

nPeers = [len(peer) for peer in peerDict.values()]
#%%
sequence = np.argsort(nPeers)
sequence = sequence[-1:0:-1]

commSize= (len(peerDict), max(nPeers))

commArray = np.zeros(commSize)
i=0
j=0

def setCommArray(commArray, peerDict, iStart,jStart):
    success = False
    for i in range(iStart+1,len(sequence)):
        me   = sequence[i]+1
        peers = peerDict[me]
        

        for j in range(jStart+1,len(peers)):
            peer = int(peers[j])
    
        
    
    
            for col in range(commArray.shape[1]):
                if commArray[me-1,j] == 0:
                    if me not in commArray[:,col] and peer not in commArray[:,col]:
                        #set communication
                        commArray[me-1,col] = peer
                        print peer, col
                        
                        commArray[peer-1,col] = me
                        success = True
            if success:
                commArray = setCommArray(commArray, peerDict, i, j)
                

    return commArray
            
commArray = setCommArray(commArray, peerDict, 0, -1)    
    