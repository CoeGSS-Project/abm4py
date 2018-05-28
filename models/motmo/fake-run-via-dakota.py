#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dakota.interfacing as di
import sys
import time

sys.path = ['../../../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

from mpi4py import MPI


comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()

dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])

wakeuptime=time.time()+1000 #start again exactly 20 seconds from now
while (time.time()<wakeuptime): 
    pass


dakotaResults['rank'].function = mpiRank
dakotaResults.write()
