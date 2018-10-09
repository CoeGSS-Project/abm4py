#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################ //{ Copyright
##
##    Copyright (c) 2018 Global Climate Forum e.V. 
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################ //}

# Generates a dakota input file from a template
# needs 3 arguments:
# 1: fileName of template (dakota input file)
# 2: configuration fileName (python file)
# 3: fileName of generated dakota input file

import io
import sys

responses = modelFileName = responsesScript = continuousVariables = staticStrings = False
exec(open(sys.argv[2]).read())
assert(responses and modelFileName and responsesScript and continuousVariables and (staticStrings != False))

def addQuotes(s):
    return "'" + s + "' "

responseDescriptors = "'runNo' "
weights = '0 '
numResponses = 1
for o in responses:
    responseDescriptors += addQuotes('o_' + o[0])
    numResponses += 1
    weights += str(o[1]) + ' '

continuousDescriptors = continuousInitialPoint = continuousLowerBounds = continuousUpperBounds = ''
for u in continuousVariables:
    continuousDescriptors += addQuotes(u[0])
    continuousInitialPoint += str(u[1]) + ' '
    continuousLowerBounds += str(u[2]) + ' '
    continuousUpperBounds += str(u[3]) + ' '
    
# static strings
print(responsesScript)
staticStrings['calcResponsesScript'] = responsesScript
staticStrings['modelFileName'] = modelFileName
staticStringDescriptors = staticStringElements = ''
for sk in staticStrings.keys():
    staticStringDescriptors += addQuotes(sk)

for sv in staticStrings.values():
    staticStringElements += addQuotes(sv)
    
# replace     
with io.open(sys.argv[1], mode = 'r') as template:
    with io.open(sys.argv[3], mode = 'w') as out:
        for line in template:
            line = line.replace('%NUM_STATIC_STRINGS%', str(len(staticStrings)))
            line = line.replace('%STATIC_STRING_DESCRIPTORS%', staticStringDescriptors)
            line = line.replace('%STATIC_STRING_ELEMENTS%', staticStringElements)

            line = line.replace('%NUM_CONTINUOUS%', str(len(continuousVariables)))
            line = line.replace('%CONTINUOUS_DESCRIPTORS%', continuousDescriptors)
            line = line.replace('%CONTINUOUS_INITIALPOINT%', continuousInitialPoint)
            line = line.replace('%CONTINUOUS_LOWERBOUNDS%', continuousLowerBounds)
            line = line.replace('%CONTINUOUS_UPPERBOUNDS%', continuousUpperBounds)

            line = line.replace('%NUM_UNIFORM%', str(len(continuousVariables)))
            line = line.replace('%UNIFORM_DESCRIPTORS%', continuousDescriptors)
            line = line.replace('%UNIFORM_LOWERBOUNDS%', continuousLowerBounds)
            line = line.replace('%UNIFORM_UPPERBOUNDS%', continuousUpperBounds)

            line = line.replace('%NUM_RESPONSES%', str(numResponses))
            line = line.replace('%RESPONSE_DESCRIPTORS%', responseDescriptors)
            line = line.replace('%RESPONSE_WEIGHTS%', weights)

            line = line.replace('%MODEL_FILENAME%', modelFileName)
            out.write(line)
