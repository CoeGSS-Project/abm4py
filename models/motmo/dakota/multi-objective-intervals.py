def calcResults(earth, dakotaResults):
    years = range(2012, 2018)  # 2018 is not included
    regionIds = earth.getParameters('regionIDList')

    def inInterval(data, real, lowFactor, highFactor):
        dataL = data * lowFactor
        dataH = data * highFactor
        return (dataL < real < dataH) * -1

    for year in years:
        for regionId in regionIds:
            r = str(regionId)
            y = str(year)
            stocksRec = earth.globalRecord['stock_' + r]
            # the stockData is a dict with the timeIndex (not the year) as key, and
            # the list with the values of the different mobilityTypes as value
            stocksData = stocksRec.calDataDict
            step = earth.year2step(year)
            dakotaResults['o_numCombCars_' + r + '_' + y].function = stocksRec.rec[step, 0]
            dakotaResults['o_numElecCars_' + r + '_' + y].function = stocksRec.rec[step, 1]
            dakotaResults['o_dataCombCars_' + r + '_' + y].function = stocksData[step][0]
            dakotaResults['o_dataElecCars_' + r + '_' + y].function = stocksData[step][1]
            dakotaResults['o_inIntervalComb_' + r + '_' + y].function = inInterval(stocksData[step][0] / 10,
                                                                                   stocksRec.rec[step, 0],
                                                                                   0.8,
                                                                                   1.2)
            dakotaResults['o_inIntervalElec_' + r + '_' + y].function = inInterval(stocksData[step][1] * 3,
                                                                                   stocksRec.rec[step, 1],
                                                                                   0.8,
                                                                                   1.2)
