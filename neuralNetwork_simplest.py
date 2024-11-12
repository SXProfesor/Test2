
from __future__ import division
__author__ = 'sxubach'

import cmath
import random
import numpy as np



def sigmoid(inputs, weights, b):
    N = 0
    for j in range(0,len(inputs)):
        N += inputs[j]*weights[j]
    N += b

    output = np.real(1/(1 + cmath.exp(-N)))

    return output


def creator (I,H,O):
    H_weights = [[random.random() for i in range(I)] for j in range(H)]

    O_weights = [[random.random() for i in range(H)] for j in range(O)]

    return H_weights,O_weights


def fordward_NN(H_weights,O_weights,I_inputs):
    I = len(I_inputs)
    H = len(H_weights)
    O = len(O_weights)

    I_outputs = [0]*I
    H_outputs = [0]*H
    O_outputs = [0]*O

    for i in range(0,I):
        I_outputs[i] = I_inputs[i]

    for i in range(0,H):
        H_outputs[i] = sigmoid(I_outputs,H_weights[i],0.35)

    for i in range(0,O):
        O_outputs[i] = sigmoid(H_outputs,O_weights[i],0.6)


    return O_outputs,H_outputs,I_outputs


I=2
H=4
O=1
tolerance = 0.001
learningRate = 0.5

H_weights, O_weights = creator(I, H, O)

'''
print 'Fordward NN:'
Ooutputs,Houtputs,Ioutputs = fordward_NN(H_weights,O_weights,[[0.05], [0.1]])
print Ooutputs
'''

#training data:
training = ([[0,0], [0]],  [[0,1], [0]],  [[1,0], [0]],  [[1,1], [1]])


error_F = 10
epoc = 0
while(error_F>tolerance):

    epoc += 1
    error_F = 0
    error = [0]*len(training)
    Output_F = [[0]*O] * len(training)

    for j in range(0,len(training)):

        O_delta = [0] * O
        O_aux = [0] * O
        H_delta = [0] * H
        H_aux = [0] * H
        O_deriv = [0] * O
        H_deriv = [0] * H

        O_outputs, H_outputs, I_outputs = fordward_NN(H_weights, O_weights, training[j][0])
        Output_F[j] = O_outputs

        #calculating delta out
        for k in range(0,O):
            #print 'target output: ' + str(training[j][1][k]) + ' Current output: ' + str(O_outputs[k])
            error[j] = -(training[j][1][k]-np.real(O_outputs[k]))
            O_aux[k] = O_outputs[k] * (1 - O_outputs[k])
            O_delta[k] = error[j] * O_aux[k]

        #calculating delta hidden
        for h in range(0,H):
            H_error = 0
            for k in range(0,O):
                H_error += O_weights[k][h] * O_delta[k]
            H_aux[h] = H_outputs[h] * (1 - H_outputs[h])
            H_delta[h] = H_aux[h] * H_error

        #update weights out
        for k in range(0,O):
            for h in range(0,H):
                O_deriv[k] = O_delta[k] * H_outputs[h]
                O_weights[k][h] -= learningRate*O_deriv[k]

        #update weights hidden
        for h in range(0,H):
            for i in range(0,I):
                H_deriv[h] = H_delta[h] * I_outputs[i]
                H_weights[h][i] -= H_deriv[h]*learningRate

        error_F += 0.5*error[j]**2


print('Exiting in Epoc: ' + str(epoc) + ' with error: ' + str(error_F))
for t in range(0,len(training)):
    print('target output: ' + str(training[t][1]) + ' Current output: ' + str(Output_F[t]))