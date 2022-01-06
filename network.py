from numpy.random import default_rng
from math import e
import json

rng = default_rng( 1 )

class Network:
    def __makeLayer( self, nInputs, nNeurons, activationType, verbose = False ):
        weights = [ [ ( rng.random() - 0.5 ) / 2 for nInput in range( nInputs ) ] for nNeuron in range( nNeurons ) ]
        biases = [ ( rng.random() - 0.5 ) / 2 for nNeuron in range( nNeurons ) ]
        activation = activationType

        if verbose:
            print( "\n", "Weights:", weights )
            print( "Biases:", biases )
            print( "Activation:", activation )

        return { "weights": weights, "biases": biases, "activation": activation }
    
    def __init__( self, dir = None, nInputs = 1, nOutputs = 1, nLayers = 1, nNeurons = 1, verbose = False ):
        if dir == None:
            self.structure = { "nInputs" : nInputs,
                            "nOutputs" : nOutputs,
                            "nLayers" : nLayers,
                            "nNeurons" : nNeurons }

            # Make layers weights, biases, and activations
            self.layers = []
            if nLayers > 1:
                self.layers.append( self.__makeLayer( nInputs = nInputs, nNeurons = nNeurons, activationType = "relu", verbose = verbose ) )
                for n in range( nLayers - 2 ):
                    self.layers.append( self.__makeLayer( nInputs = nNeurons, nNeurons = nNeurons, activationType = "relu", verbose = verbose ) )
                self.layers.append( self.__makeLayer( nInputs = nNeurons, nNeurons = nOutputs, activationType = "softmax", verbose = verbose ) ) 
            else:
                self.layers.append( self.__makeLayer( nInputs = nInputs, nNeurons = nOutputs, activationType = "softmax", verbose = verbose ) )
        else:
            with open( dir ) as file:
                network = json.load( file )
                self.structure = network[ "structure" ]
                self.layers = network[ "layers" ]

    def save( self, name, dir = None ):
        if dir == None:
            dir = name
        else:
            dir = dir + name    

        with open( dir + ".json", "w" ) as file:
            json.dump( { "structure" : self.structure, "layers" : self.layers }, file, indent = 4 )

    def __step_forward( self, inputs, nLayer ):
        outputs = []
        for nNeuron in range( len( self.layers[ nLayer ][ "weights" ] ) ):
            outputs.append( 0 )
            for input, weight in zip( inputs, self.layers[ nLayer ][ "weights" ][ nNeuron ] ):
                outputs[ nNeuron ] += input * weight
            outputs[ nNeuron ] += self.layers[ nLayer ][ "biases" ][ nNeuron ]

        # ReLU activation
        if self.layers[ nLayer ][ "activation" ] == "relu":
            for nOutput in range( len( outputs ) ):
                if outputs[ nOutput ] < 0: 
                    outputs[ nOutput ] = 0
        
        # Softmax activation
        else:
            outputs = [ e ** output for output in outputs ]

            sum = 0
            for output in outputs:
                sum += output

            outputs = [ output / sum for output in outputs ]

        return outputs

    def __step_backward( self, nLayer, inputs ):
        if nLayer > 0:
            inputs = self.__step_forward( inputs, 0 )
            for nLayer_temp in range( 1, nLayer ):
                inputs = self.__step_forward( inputs, nLayer_temp )
 
        dWeights = []
        for nNeuron in range( len( self.layers[ nLayer ][ "weights" ] ) ):
            dWeights.append( [] )
            print( inputs )
            print( self.layers[ nLayer ][ "weights" ][ nNeuron ] )
            for input, weight in zip( inputs, self.layers[ nLayer ][ "weights" ][ nNeuron ] ):
                if self.layers[ nLayer ][ "activation" ] == "relu": 
                    if input > 0: activationDerivative = 1
                    else: activationDerivative = 0
                    dWeights[ nNeuron ].append( activationDerivative * input * weight )
                else:
                    if input > 0: activationDerivative = 1
                    else: activationDerivative = 0
                    dWeights[ nNeuron ].append( activationDerivative * input * weight )
        
        return dWeights

    def __updateWeights( self, dWeights, nLayer ):
        for nNeuron in range( len( self.layers[ nLayer ][ "weights" ] ) ):
            for nWeight in range( len( self.layers[ nLayer ][ "weights" ][ nNeuron ] ) ):
                # print( self.layers[ nLayer ][ "weights" ][ nNeuron ][ nWeight ] )
                # print( dWeights[ nNeuron ][ nWeight ] )
                self.layers[ nLayer ][ "weights" ][ nNeuron ][ nWeight ] += dWeights[ nNeuron ][ nWeight ]

    def __updateBiases( self, dBiases, nLayer ):
        for nBias in range( len( self.biases ) ):
            self.layers[ nLayer ][ nBias ] += dBiases[ nBias ]

    def forward( self, input ):
        output = self.__step_forward( input, 0 )
        if len( self.layers ) > 1:
            for nLayer in range( 1, len( self.layers ) ):
                output = self.__step_forward( output, nLayer )

        return output

    def backward( self, input, input_oneHot ):
        for nLayer in range( len( self.layers ) - 1, -1, -1 ):
            dWeights = self.__step_backward( nLayer, input )
            self.__updateWeights( dWeights, nLayer ) 
            # print( "\n", dWeights)

def backwardPropagation( layers, inputData, inputData_oneHot ):
    for nLayer in range( len( layers ), 0, -1 ):
        layers[ nLayer ].backward( inputData )
        layers[ nLayer ].updateWeights( layers[ nLayer ].dWeights )

def __getLoss( predictedOutputs, actualOutputs ):
    loss = 0
    for predictedOutput, actualOutput in zip( predictedOutputs, actualOutputs ):
        for pOutput, aOutput in zip( predictedOutput, actualOutput ):
            loss += ( aOutput - pOutput ) ** 2

    return loss