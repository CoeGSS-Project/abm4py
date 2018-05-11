def prioritiesCalibrationTest():

    householdSetup(earth, parameters, calibration=True)
    df = pd.DataFrame([],columns=['prCon','prEco','prMon','prImi'])
    for agID in earth.nodeDict[3]:
        df.loc[agID] = earth.graph.vs[agID]['preferences']

    propMat = np.array(np.matrix(earth.graph.vs[earth.nodeDict[3]]['preferences']))

    return earth


def setupHouseholdsWithOptimalChoice():

    householdSetup(earth, parameters)
    initMobilityTypes(earth, parameters)
    #earth.market.setInitialStatistics([500.0,10.0,200.0])
    for household in earth.iterEntRandom(HH):
        household.takeAction(earth, household.adults, np.random.randint(0,earth.market.nMobTypes,len(household.adults)))

    for cell in earth.iterEntRandom(CELL):
        cell.step(earth.market.kappa)

    earth.market.setInitialStatistics([1000.,5.,300.])

    for household in earth.iterEntRandom(HH):
        household.calculateConsequences(earth.market)
        household.util = household.evalUtility()

    for hh in iter(earth.nodeDict[HH]):
        oldEarth = copy(earth)
        earth.entDict[hh].bestMobilityChoice(oldEarth,forcedTryAll = True)
    return earth
