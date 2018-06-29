#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

This file is part on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt

def shrink(data, rows, cols):
    print(data.shape[0]/rows, data.shape[1]/cols)
    return data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols)).sum(axis=1).sum(axis=2)

population = np.load('pop_count.npy')
population = population[5:245,10:280]

newPop = shrink(population, 48, 54)

plt.clf()
plt.imshow(newPop)
plt.colorbar()
print(np.sum(~np.isnan(newPop)))
np.save('coarse_pop_count.npy', newPop)