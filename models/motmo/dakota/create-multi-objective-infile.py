#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generates a dakota input file from a template
# needs 3 arguments:
# 1: fileName of template (dakota input file)
# 2: configuration fileName (python file)
# 3: fileName of generated dakota input file

import io
import sys

regionIds = outputValues = years = scenarioFileName = resultScript = continuousVariables = calcWeight = False
exec(open(sys.argv[2]).read())
assert(regionIds and outputValues and years and scenarioFileName and resultScript and continuousVariables and calcWeight)

def addQuotes(s):
    return "'" + s + "' "

responseDescriptors = "'runNo' "
weights = '0 '
numResponses = 1
for o in outputValues:
    for r in regionIds:
        for y in years:
            responseDescriptors += addQuotes(o + '_' + str(r) + '_' + str(y))
            numResponses += 1
            weights += str(calcWeight(o, r, y)) + ' '

continuousDescriptors = continuousInitialPoint = continuousLowerBounds = continuousUpperBounds = ''
for u in continuousVariables:
    continuousDescriptors += addQuotes(u[0])
    continuousInitialPoint += str(u[1]) + ' '
    continuousLowerBounds += str(u[2]) + ' '
    continuousUpperBounds += str(u[3]) + ' '


    
with io.open(sys.argv[1], mode = 'r') as template:
    with io.open(sys.argv[3], mode = 'w') as out:
        for line in template:
            line = line.replace('%NUM_RESPONSES%', str(numResponses))
            line = line.replace('%RESPONSE_DESCRIPTORS%', responseDescriptors)
            line = line.replace('%RESPONSE_WEIGHTS%', weights)
            line = line.replace('%SCENARIO_FILENAME%', addQuotes(scenarioFileName))
            line = line.replace('%RESULT_SCRIPT%', addQuotes(resultScript))
            line = line.replace('%NUM_CONTINUOUS%', str(len(continuousVariables)))
            line = line.replace('%CONTINUOUS_DESCRIPTORS%', continuousDescriptors)
            line = line.replace('%CONTINUOUS_INITIALPOINT%', continuousInitialPoint)
            line = line.replace('%CONTINUOUS_LOWERBOUNDS%', continuousLowerBounds)
            line = line.replace('%CONTINUOUS_UPPERBOUNDS%', continuousUpperBounds)
            out.write(line)
