
import sys, os
from os.path import expanduser
home = expanduser("~")
#sys.path.append(home + '/python/decorators/')
sys.path.append(home + '/python/modules/')
sys.path.append(home + '/python/agModel/modules/')

import socket
dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = [dir_path + '/h5py/build/lib.linux-x86_64-2.7'] + sys.path 
    sys.path = [dir_path + '/mpi4py/build/lib.linux-x86_64-2.7'] + sys.path 
else:
    import matplotlib
    matplotlib.use('Agg')    
#from deco_util import timing_function
import numpy as np
import time
#import mod_geotiff as gt # not working in poznan

from mpi4py import  MPI
import h5py
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

#from para_class_mobilityABM import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
#import class_auxiliary  as aux #convertStr

import matplotlib.pylab as plt
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
#import matplotlib.pyplot as plt

import pandas as pd
from bunch import Bunch
from copy import copy
import csv
from scipy import signal

comm.Barrier()
if comm.rank == 0:
    print 'done'