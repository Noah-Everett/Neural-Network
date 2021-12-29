import network as code_network
import input as code_input

# Get data
# inputs, inputs_classifications, inputs_oneHot = code_input.getData( dataSets = [ "Quest", "ON" ], dir = "/Users/noaheverett/Documents/Codes/Neural Network/Data Sets/" )
inputs = [ [ n for n in range( 10 ) ], [ n for n in range( 10 ) ] ]
inputs_classifications = [ "Quest" for nInput in inputs ]
inputs_oneHot = [ 0 for nInput in inputs ]

# Trim input data
del inputs[ 1: ]
del inputs_classifications[ 1: ]
del inputs_oneHot[ 1: ]

# Make network
network = code_network.Network( nInputs = len( inputs[ 0 ] ), nOutputs = 3, nLayers = 3, nNeurons = 4, verbose = False ]
network.save( name = "Protein_Powder_Containers_Neural_Network", dir = "/Users/noaheverett/Documents/Codes/Neural Network/" )

# nRuns = 1
# for nRun in range( nRuns ):
    # Forward propagation
output = []
for nInput in range( len( inputs ) ): 
    output.append( network.forward( input = inputs[ nInput ] ) )
print( "\n" + "Output:", output )

    # loss = code_network.getLoss( output, inputs_oneHot )
    # print( "\n" + "Loss:", loss, "\n" )

    # Backward propagation
    # for data in inputData: 
    #     code_network.backwardPropagation( layers = layers, inputData = data, inputData_oneHot = inputData_oneHot )

