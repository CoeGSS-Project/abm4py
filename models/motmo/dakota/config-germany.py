import pandas as pd

_data = pd.read_csv('../resources_ger/calDataCV.csv', index_col=0, header=1)
regionIds = [s.split('_', 1)[1] for s in _data.index[1:]]

years = range(2012, 2018) #2018 is not included

scenarioFileName = 'parameters_ger.csv'

resultScript = 'multi-objective-intervals.py'

outputValues = ['o_numCombCars', 'o_numElecCars']

continuousVariables = [
    ('innoPriority', 0.2, 0.0, 0.5),
    ('mobIncomeShare', 0.2, 0.1, 0.4)
]

def calcWeight(var, region, year):
    return 1
