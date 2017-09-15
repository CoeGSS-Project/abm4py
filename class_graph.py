#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:31:00 2017

@author: Andreas Geiges, Global Climate Forum e.V.
"""

from igraph import Graph


class WorldGraph(Graph):
    """
    World graph is an agumented version of igraphs Graph that overrides some 
    functions to work with the restrictions and functionality of world.
    It ensures that for example that deleting and adding of edges are recognized
    and cached.
    It also extends some of the functionalities of igraph (e.g. add_edges).
    """
    
    def __init__(self, world, directed=None):

         Graph.__init__(self, directed=directed) 
         self.world = world
    
    def add_edges(self, edgeList, **argProps):
        
        eStart = self.ecount()
        Graph.add_edges(self, edgeList)
        for key in argProps.keys():
            self.es[eStart:][key] = argProps[key]
        

if __name__ == "__main__":
    
    world = dict()
    
    graph = WorldGraph(world)
    
    graph.add_vertices(5)
    
    graph.add_edges([(1,2),(2,3)], type=[1,1], weig=[0.5,0.5])
