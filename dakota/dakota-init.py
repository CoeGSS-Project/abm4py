import dakota.interfacing as di

dakotadir = os.getcwd()
dakotaParams, dakotaResults = di.read_parameters_file(sys.argv[1], sys.argv[2])

modelFileName = dakotaParams['modelFileName']
modelDir, modelFile = os.path.split(modelFileName)
os.chdir(modelDir)

dakotaRunNo = random.randrange(2 ** 63)

parameters = dict()
for d in dakotaParams.descriptors:
    parameters[d] = dakotaParams[d]
core.dakota.overwriteParameters(dakotaRunNo, parameters)

def reportResponse(attr, value):
    dakotaResults['o_' + attr].function = value

def finishDakota():
    if core.mpiRank == 0:
        os.chdir(dakotadir)
        exec(open(dakotaParams['calcResponsesScript']).read(), globals())
        calcResponses()
        dakotaResults['runNo'].function = int(dakotaRunNo)
        
        dakotaResults.write()

