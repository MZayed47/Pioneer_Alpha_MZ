# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:01:50 2021

@author: Mashrukh
"""


from random import seed
from random import random


 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
seed(1)

network = initialize_network(2, 4, 1)

c = 0
for layer in network:
    print("Weights from Layer " + str(c) + " : " + str(layer) + "\n")
    c += 1



# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation



# Transfer neuron activation using ReLU
def transfer(activation):
    return max(0, activation)



# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs



# test forward propagation with input pattern [1,0]
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}, {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]}, {'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349]}, {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337]}],
           [{'weights': [0.762280082457942, 0.0021060533511106927, 0.4453871940548014, 0.7215400323407826, 0.22876222127045265]}]
           ]

row = [1, 0, None]
output = forward_propagate(network, row)

print("Output Layer Neurons: " + str(output) + "\n")



##############################################################################
##############################################################################

print("#######################################################\n")



# Calculate the derivative of an output neuron (which uses sigmoid activation function)
def transfer_derivative(output):
	return output * (1.0 - output)



# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



# test backpropagation of error
network = [[{'output': 0.6213859615555266, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}, {'output': 0.762280082457942, 'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]}, {'output': 0.0021060533511106927, 'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349]}, {'output': 0.4453871940548014, 'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337]}],
           [{'output': 0.7215400323407826, 'weights': [0.762280082457942, 0.0021060533511106927, 0.4453871940548014, 0.7215400323407826, 0.22876222127045265]}]
           ]


# Check the outputs and backward weights before calling error function
for i in reversed(range(len(network))):
    layer = network[i]
    print("Backward weights from Layer " + str(i+1) + " with length " + str(len(layer)) + " : " + str(layer) + "\n")


print("#######################################################\n")

expected = [0, 1]

backward_propagate_error(network, expected)


d=0
for layer in network:
    d += 1
    print("Output, Weights, & error of Backward Layer " + str(d) + ' : ' + str(layer) + "\n")



