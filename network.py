from numpy.random import default_rng
from math import e
import os
import json

import misc

rng = default_rng( 0 )

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
    
    def __saveNetwork( self, file ):
        file.write( "Network Structure: nInputs: " + str( self.nInputs ) + "\n" )
        file.write( "                   nOutputs: " + str( self.nOutputs ) + "\n" )
        file.write( "                   nLayers: " + str( self.nLayers ) + "\n" )
        file.write( "                   nNeurons: " + str( self.nNeurons ) + "\n\n" )



        for nLayer in range( len( self.layers ) ):
            file.write( "Layer " + misc.pad( nLayer, len( self.layers ) - 1 ) + ": Weights: " + str( self.layers[ nLayer ][ "weights" ][ 0 ][ 0 ] ) + "\n" )
            for nNeuron in range( len( self.layers[ nLayer ][ "weights" ] ) ):
                print( len( self.layers[ nLayer ][ "weights" ][ nNeuron ] ) )
                for nWeight in range( len( self.layers[ nLayer ][ "weights" ][ nNeuron ] ) ):
                    if nNeuron != 0 or nWeight != 0:
                        file.write( " " * ( 17 + len( misc.pad( nLayer, len( self.layers ) - 1 ) ) ) + str( self.layers[ nLayer ][ "weights" ][ nNeuron ][ nWeight ] ) + "\n" )
            file.write( " " * ( 8 + len( misc.pad( nLayer, len( self.layers ) - 1 ) ) ) + "Biases: " + str( self.layers[ nLayer ][ "biases" ][ 0 ] ) + "\n" )
            for nNeuron in range( len( self.layers[ nLayer ][ "biases" ] ) ):
                if nNeuron != 0:
                    file.write( " " * ( 16 + len( misc.pad( nLayer, len( self.layers ) - 1 ) ) ) + str( self.layers[ nLayer ][ "biases" ][ nNeuron ] ) + "\n" )
            file.write( " " * ( 8 + len( misc.pad( nLayer, len( self.layers ) - 1 ) ) ) + "Activation: " + self.layers[ nLayer ][ "activation" ] + "\n\n" )

    def __init__( self, nInputs, nOutputs, nLayers, nNeurons, verbose = False ):
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

    def save( self, name, dir ):        
        with open( dir + name + ".json", "w" ) as file:
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

    def __step_backward( self, input ):
        dWeights = []
        for nNeuron in range( len( self.weights ) ):
            dWeights.append( [] )
            for input, weight in zip( input, self.weights[ nNeuron ] ):
                if self.activation == "relu": 
                    if input > 0: activationDerivative = 1
                    else: activationDerivative = 0
                    dWeights[ nNeuron ].append( activationDerivative * input * weight )
                # elif self.activation == "softmax":

    def forward( self, input ):
        output = self.__step_forward( input, 0 )
        if len( self.layers ) > 1:
            for nLayer in range( 1, len( self.layers ) ):
                output = self.__step_forward( output, nLayer )

        return output

    def backward( self, layers, inputData, inputData_oneHot ):
        for nLayer in range( len( layers ), 0, -1 ):
            layers[ nLayer ].backward( inputData )
            layers[ nLayer ].updateWeights( layers[ nLayer ].dWeights )

    def updateWeights( self, dWeights ):
        for nNeuron in range( len( self.weights ) ):
            for nWeight in range( len( self.weights[ 0 ] ) ):
                self.weights[ nNeuron ][ nWeight ] += dWeights[ nNeuron ][ nWeight ]

    def updateBiases( self, dBiases ):
        for nBias in range( len( self.biases ) ):
            self.biases[ nBias ] += dBiases[ nBias ]



def backwardPropagation( layers, inputData, inputData_oneHot ):
    for nLayer in range( len( layers ), 0, -1 ):
        layers[ nLayer ].backward( inputData )
        layers[ nLayer ].updateWeights( layers[ nLayer ].dWeights )

def getLoss( predictedOutputs, actualOutputs ):
    loss = 0
    for predictedOutput, actualOutput in zip( predictedOutputs, actualOutputs ):
        for pOutput, aOutput in zip( predictedOutput, actualOutput ):
            loss += ( aOutput - pOutput ) ** 2

    return loss