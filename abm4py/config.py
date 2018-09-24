#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, Global Climate Forun e.V. (GCF)
http://www.globalclimateforum.org

This file is part of ABM4py.

ABM4py is free software: you can redistribute it and/or modify it 
under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, version 3 only.

ABM4py is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>. 
GNU Lesser General Public License version 3 (see the file LICENSE).
"""


import numpy as np

################################### CONFIG ###################################

# This compound typtes are allowed as agent attributes (agent state)
ALLOWED_MULTI_VALUE_TYPES = (list, tuple, np.ndarray)

# This defines the type if intergers that IDs and global IDs are of.
GID_TYPE = np.int64
ID_TYPE = np.int32

# this is the inital size of hte node and edge attribute array (see graph.py)
GRAPH_ARRAY_INIT_SIZE = 100

# Defaulf factor for exending arrays:
EXTENTION_FACTOR = 2

MAX_NODES = int(1e6)
MAX_LINKS = int(1e6)