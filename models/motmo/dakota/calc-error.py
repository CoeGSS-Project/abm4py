def calcResults(earth, dakotaResults):
    dakotaResults["relativeError"].function = earth.globalRecord['stock_6321'].evaluateRelativeError()
    # dakotaResults["absoluteError"].function = earth.globalRecord['stock_6321'].evaluateAbsoluteError()
