modelFileName = '../examples/bass_diffusion/01_bass_network_dakota.py'

responsesScript = './response-scripts/bass-sampling.py'

# (name, initial points, lowerLimit, upperLimit)
continuousVariables = [
    ('imitation', 0.2, 0.0, 0.5),
    ('innovation', 15, 10, 20)
]

staticStrings = []

# (name, weight)
responses = [
    ('switchFraction', 1)
]
