regionIds = [ 6321 ]

years = range(2012, 2018) #2018 is not included

scenarioFileName = 'parameters_med.csv'

resultScript = 'multi-objective-intervals.py'

outputValues = [
    'o_numCombCars',
    'o_numElecCars',
    'o_dataCombCars',
    'o_dataElecCars',
    'o_inIntervalComb',
    'o_inIntervalElec'
]

continuousVariables = [
    ( 'innoPriority', 0.2, 0.0, 0.5 ),
    ( 'mobIncomeShare', 0.2, 0.1, 0.4 )
]

def calcWeight(var, region, year):
    if var == 'o_inIntervalElec' or var == 'o_inIntervalComb':
        return 1
    return 0

