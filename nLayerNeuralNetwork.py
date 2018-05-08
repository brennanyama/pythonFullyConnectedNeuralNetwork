# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:43:35 2018

@author: brenn
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 09:51:35 2018

@author: Brennan

Much mahalo to help from Milo Spencer-Harper
    https://github.com/miloharper/simple-neural-network/blob/master/main.py
    https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py
    
This is a fully-connected neural network, i.e. the number of inputs to each
neuron are equal to the number of neurons from the previous layer.  There can
be any number of neurons per layer, and any number of layers.

"""

#from numpy import exp, array, random, dot

import numpy as np

class NeuronLayer():
    def __init__(self, nNeurons, nNeuronInputs):
        self.synapticWeights = 2*np.random.random((nNeuronInputs,nNeurons))-1
        self.outputs = np.array([1,1])
        self.error = np.array([1,1])
        self.delta = np.array([1,1])

class NeuralNetwork():
    
    def __init__(self, networkLayout):
        # networkLayout: [n x 1] matrix where each value is the number of neurons in that respective layer
        # First value MUST match features of input, and last value MUST match features of output.
        self.networkLayout = networkLayout
        self.nInputFeatures = networkLayout[0]
        self.nOutputFeatures = networkLayout[-1]
        self.nLayers = len(self.networkLayout)
        self.layers = []
    
    def makeNeuronLayers(self):
        # Create list of neuron layers based on user input
        for i in range(self.nLayers):
            if i == 0:
                self.layers.append(NeuronLayer(self.networkLayout[i],self.nInputFeatures))
            else:
                self.layers.append(NeuronLayer(self.networkLayout[i],self.networkLayout[i-1]))
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, trainingInput, trainingOutput, trainingIterations):
        for i in range(trainingIterations):
            
            # Pass the training set through our neural network
            self.predict(trainingInput)
                        
            # Calculate error for each layer (starting from final layer, working back)
            for j in range(self.nLayers):

                if j == 0:
                    self.layers[self.nLayers-(j+1)].error = (trainingOutput - self.layers[self.nLayers-(j+1)].outputs)
                    self.layers[self.nLayers-(j+1)].delta = np.multiply(self.layers[self.nLayers-(j+1)].error,self.__sigmoid_derivative(self.layers[self.nLayers-(j+1)].outputs))
                    
                else:
                    self.layers[self.nLayers-(j+1)].error = np.dot(self.layers[self.nLayers-(j)].delta,self.layers[self.nLayers-(j)].synapticWeights.T)
                    self.layers[self.nLayers-(j+1)].delta = np.multiply(self.layers[self.nLayers-(j+1)].error,self.__sigmoid_derivative(self.layers[self.nLayers-(j+1)].outputs))
                
            # Adjust weights (starting from the first layer, working forward)
            for j in range(self.nLayers):

                if j == 0:
                    adjust = np.dot(trainingInput.T,self.layers[j].delta)
                    self.layers[j].synapticWeights += adjust

                else:
                    adjust = np.dot(self.layers[j-1].outputs.T,self.layers[j].delta)
                    self.layers[j].synapticWeights += adjust


    # One prediction cycle for the neural network
    def predict(self, input):
            
        for i in range(self.nLayers):
            if i == 0:
                self.layers[i].outputs = (self.__sigmoid(np.dot(input,self.layers[i].synapticWeights)))
            
            else:
                self.layers[i].outputs = (self.__sigmoid(np.dot(self.layers[i-1].outputs,self.layers[i].synapticWeights)))
            
            outputLayer = self.layers[self.nLayers-1].outputs

        return outputLayer
    
    def print_weights(self):
        
        for i in range(self.nLayers):
            print(["layer",i+1])
            print(self.layers[i].synapticWeights)
        print("")

if __name__ == "__main__":

    nRuns = 20
    for i in range(nRuns):
    
        # Seed the random number generator
        np.random.seed(i)
    
        # Create neural network
        # networkLayout [n x 1] where; n = number of neurons in that respective layer (last number MUST be 1)
        nn1 = NeuralNetwork([3,6,6,6,6,6,6,1])
    
        # Form neuron layers for network
        nn1.makeNeuronLayers()
            
        # Train
#        trainingInputs = np.array([[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0],[1,0,0]])
#        trainingOutputs = np.array([[0],[1],[0],[1],[0],[0]])
        trainingInputs = np.array([[0,0,1],[0,1,0],[1,0,1],[0,1,1],[0,0,0],[1,1,0],[1,1,1]])
        trainingOutputs = np.array([[0],[1],[1],[0],[1],[0],[1]])
        
        # Train    
#        print('annA.'+str(i+1)+'= [')
        nn1.train(trainingInputs,trainingOutputs,5)
        output = nn1.predict(np.array([[1,0,0]]))
        print('ann.G'+str(i+1)+'=['+str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,10)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,50)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')    
        nn1.train(trainingInputs,trainingOutputs,100)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,500)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,1000)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,5000)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,10000)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+';')
        nn1.train(trainingInputs,trainingOutputs,50000)
        output = nn1.predict(np.array([[1,0,0]]))
        print(str(output)[2:-2]+'];')
    
