#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generates a dakota input file from a template
# needs 3 arguments:
# 1: template fileName
# 2: configuration fileName
# 3: fileName of generated file

import io
import sys

regionIds = outputValues = years = scenarioFileName = resultScript = uniformVariables = False
exec(open(sys.argv[2]).read())
assert(regionIds and outputValues and years and scenarioFileName and resultScript and uniformVariables)

def addQuotes(s):
    return "'" + s + "' "

responseDescriptors = ''
numResponses = 0
for o in outputValues:
    for r in regionIds:
        for y in years:
            responseDescriptors += addQuotes(o + '_' + str(r) + '_' + str(y))
            numResponses += 1

uniformDescriptors = uniformLowerBounds = uniformUpperBounds = ''
for u in uniformVariables:
    uniformDescriptors += addQuotes(u[0])
    uniformLowerBounds += str(u[1]) + ' '
    uniformUpperBounds += str(u[2]) + ' '

with io.open(sys.argv[1], mode = 'r') as template:
    with io.open(sys.argv[3], mode = 'w') as out:
        for line in template:
            line = line.replace('%NUM_RESPONSES%', str(numResponses))
            line = line.replace('%RESPONSE_DESCRIPTORS%', responseDescriptors)
            line = line.replace('%SCENARIO_FILENAME%', addQuotes(scenarioFileName))
            line = line.replace('%RESULT_SCRIPT%', addQuotes(resultScript))
            line = line.replace('%NUM_UNIFORM%', str(len(uniformVariables)))
            line = line.replace('%UNIFORM_DESCRIPTORS%', uniformDescriptors)
            line = line.replace('%UNIFORM_LOWERBOUNDS%', uniformLowerBounds)
            line = line.replace('%UNIFORM_UPPERBOUNDS%', uniformUpperBounds)
            out.write(line)
