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
        gID = self.getGlobID(world)
        self.attr['gID'] = gID
        self.gID = gID
         #kwProperties['gID'] = self.gID
        
        #self.attr['gID'] = self.gID
        
        # if nID is -1, the agent is generated emtpy and receives its properties
        # and IDs later by mpi communication
#        if nID is not -1:
#            self.nID = nID
#            self.attr, self.dataID = self._graph.getNodeView(nID)
#            self.mpiOwner =  int(world.mpiRank)
#            self.mpiOwner = int(world.mpiRank)
        self.mpiPeers = list()

    def getGlobID(self,world):
        return next(world.globIDGen)
# from location
#    def registerChild(self, world, entity, liTypeID=None):
#        world.addLink(liTypeID, self.nID, entity.nID)
#        entity.loc = self
#
#        if len(self.mpiPeers) > 0: # node has ghosts on other processes
#            for mpiPeer in self.mpiPeers:
#                #print 'adding node ' + str(entity.nID) + ' as ghost'
#                agTypeID = world.graph.class2NodeType(entity.__class__)
#                world.papi.queueSendGhostNode( mpiPeer, agTypeID, entity, self)
#
#        return self.mpiPeers

    def registerChild(self, world, entity, liTypeID):
        """
        
        """
        # why is register child also adding a link?
        if liTypeID is not None:
            world.addLink(liTypeID, self.nID, entity.nID)

        if len(self.mpiPeers) > 0: # node has ghosts on other processes
            for mpiPeer in self.mpiPeers:
                #print 'adding node ' + str(entity.nID) + ' as ghost'
                agTypeID = world.graph.class2NodeType(entity.__class__)
                world.papi.queueSendGhostNode( mpiPeer, agTypeID, entity, self)

        return self.mpiPeers


class Neighborhood():
    """
    This enhancement allows agents to iterate over neigboring instances and thus
    funtion as collectives of agents. 
    """
    def __init__(self, world, nID = -1, **kwProperties):
        self.__getAgent = world.getAgent
        self.Neighborhood = dict()
    
    def getNeighbor(self, peerID):
        return self.__getAgent(nodeID=peerID)
    
    def reComputeNeighborhood(self, liTypeID):
        #self.Neighborhood[liTypeID] = [self.getNeighbor(ID) for ID in self.getPeerIDs(liTypeID)]
        self.Neighborhood[liTypeID] = self.getPeerAttr('instance', liTypeID= liTypeID)
        
    def iterNeighborhood(self, liTypeID):
        return iter(self.Neighborhood[liTypeID])
        
class Mobile():
    """
    This enhancemement allows agents to move in the spatial domain. Currently
    this does not work in the parallel version
    """
    
    
    def __init__(self, world, nID = -1, **kwProperties):
        # assert that position is declated as an agents attribute, since 
        # moving relates to the 'pos' attribute
        #TODO can be made more general"
        
        if world.isParallel:
            raise(BaseException('Mobile agents are not working in parallel'))
            
        assert 'pos' in kwProperties.keys()
        
        self._setLocationDict(world.getLocationDict())
        
    def move(self, newX, newY, spatialLinkTypeID):
        self.attr['pos'] = [ newX, newY]
        self.remLink(friendID=self.loc.nID, liTypeID=spatialLinkTypeID)
        self.loc.remLink(self.nID, liTypeID=spatialLinkTypeID)
        
        self.loc = self.locDict[(newX, newY)]
        
        self.addLink(friendID=self.loc.nID, liTypeID=spatialLinkTypeID)
        self.loc.addLink(self.nID, liTypeID=spatialLinkTypeID)

    @classmethod
    def _setLocationDict(cls, locDict):
        """ Makes the class variable _graph available at the first init of an entity"""
        cls.locDict = locDict
                
#    def _moveNormal(self, newPosition):
#        self.attr['pos'] = newPosition
#        
#    def move(self, newPosition, spatialLinkTypeID):
#        """ not yet implemented"""
#        pass

class Collective():
    """
    This enhancement allows agents to iterate over member instances and thus
    funtion as collectives of agents. Examples could be a pack of wolf or 
    an household of persons.
    """
    def __init__(self, world, nID = -1, **kwProperties):
        self.__getAgent = world.getAgent
        self.groups = dict()
        
    def getMember(self, peerID):
        return self.__getAgent(nodeID=peerID)
    
    def iterMembers(self, peerIDs):
        return [self.__getAgent(nodeID=peerID) for peerID in peerIDs]
 
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

    def setPeerAttr(self, prop, values, liTypeID=None, agTypeID=None, force=False):
        """
        Set the attributes of all connected nodes of an specified agTypeID
        or connected by a specfic edge type
        """
        if not force:
            raise Exception
        else:
            #import warnings
            #warnings.warn('This is violating the current rules and data get lost')

            self._graph.setOutNodeValues(self.nID, liTypeID, prop, values)    