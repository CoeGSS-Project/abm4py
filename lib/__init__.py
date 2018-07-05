#!/usr/bin/env python3
# -*- coding: UTF-8-*-
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


Philosophy:

Classes are only the hull of methods around a graph node with its connections.
Entities can only alter out-connections by themselves (out-edges belong to the source node).
Entities should therefore be fully defined by their global ID and local ID and
the underlying graph that contains all properties.

Thus, as far as possible save any property of all entities in the graph

Communication is managed via the spatial location

Thus, the function "location.registerEntity" does initialize ghost copies

"""
from __future__ import absolute_import

from .World import World
from .agent import Agent, GhostAgent
from .location import Location, GhostLocation
from . import core
#        
#if __name__ == '__main__':
#    
#    pass
    