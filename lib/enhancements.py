#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:07:26 2018

@author: gcf
"""

class Parallel():
    """
    This agent enhancements adds the required methods and variables to agents
    for parallel execution of the code.
    This contains:
        - global ID: self.gID
        - 
    """
    def __init__(self, world, nID = -1, **kwProperties):
        
        # adding properties to the attributes
        kwProperties['gID'] = self.gID
        
        # if nID is -1, the agent is generated emtpy and receives its properties
        # and IDs later by mpi communication
        if nID is not -1:
            self.nID = nID
            self.attr, self.dataID = self._graph.getNodeView(nID)
            self.mpiOwner =  int(world.mpiRank)

    def getGlobID(self,world):
        return next(world.globIDGen)



    def registerChild(self, world, entity, linkTypeID):
        """
        
        """
        # why is register child also adding a link?
        if linkTypeID is not None:
            world.addLink(linkTypeID, self.nID, entity.nID)

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                nodeTypeID = world.graph.class2NodeType(entity.__class__)
                world.papi.queueSendGhostNode( mpiPeer, nodeTypeID, entity, self)

        return self.mpiPeers


class Neigbourhood():
    """
    This enhancement allows agents to iterate over neigboring instances and thus
    funtion as collectives of agents. 
    """
    def __init__(self, world, nID = -1, **kwProperties):
        self.__getNode = world.getNode
        self.neigborhood = dict()
    
    def getNeigbor(self, peerID):
        return self.__getNode(nodeID=peerID)
    
    def reComputeNeigborhood(self, linkTypeID):
        self.neigborhood[linkTypeID] = [self.getNeigbor(ID) for ID in self.getPeerIDs(linkTypeID)]
        
    def iterNeigborhood(self, linkTypeID):
        return iter(self.neigborhood[linkTypeID])
        
class Moveable():
    
    def __init__(self, world, nID = -1, **kwProperties):
        """ assert that position is declared as an agent's attribute, since 
         moving relates to the 'pos' attribute """
        #TODO can be made more general"
        # SW: warum braucht das nID?, same question for collective
        assert 'pos' in kwProperties.keys()
        
    def _moveSpatial(self, newPosition):
        pass
        
    def _moveNormal(self, newPosition):
        self.attr['pos'] = newPosition
        
    def move(self):
        """ not yet implemented"""
        pass

class Collective():
    """
    This enhancement allows agents to iterate over member instances and thus
    funtion as collectives of agents. Examples could be a pack of wolf or 
    an household of persons.
    """
    def __init__(self, world, nID = -1, **kwProperties):
        self.__getNode = world.getNode
        self.groups = dict()
        
    def getMember(self, peerID):
        return self.__getNode(nodeID=peerID)
    
    def iterMembers(self, peerIDs):
        return [self.__getNode(nodeID=peerID) for peerID in peerIDs]
 
    def registerGroup(self, groupName, members):
        self.groups[groupName] = members
        
    def iterGroup(self, groupName):
        return iter(self.groups[groupName])
    
    def join(self, groupName, agent):
        self.groups[groupName].append(agent)
        
    def leave(self, groupName, agent):
        self.groups[groupName].remove(agent)        

class SuperPowers():
    """
    This agent-enhancement allows to write attributes of connected agents
    Use carefully and not in parallel mode.
    """
    
    def __init__(self, world, nID = -1, **kwProperties):
        # check that the framework is not parallelized since the writing of 
        # attributes from other agents violates the consistency of parallel
        # execution
        assert world.isParallel == False

    def setPeerAttr(self, prop, values, linkTypeID=None, nodeTypeID=None, force=False):
        """
        Set the attributes of all connected nodes of an specified nodeTypeID
        or connected by a specfic edge type
        """
        if not force:
            raise Exception
        else:
            #import warnings
            #warnings.warn('This is violating the current rules and data get lost')

            self._graph.setOutNodeValues(self.nID, linkTypeID, prop, values)    