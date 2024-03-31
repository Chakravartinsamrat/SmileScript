import torch
import torch.nn as nn

#A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers
#Relu sigmoid func, Rectified linear unit is an activation func output the inputs if its positive else zero
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        #defines the constructor method for the neural network class, takes 3 para
        #inputsize - size opf the input features 
        #hidden size the number of units in the hidden layer
        #num classes- number of output clases
        super(NeuralNet, self).__init__()
        #super calls constructor
        self.l1 = nn.Linear(input_size, hidden_size)
        #first linear transformation layer, input-input size output-hiddenlayer y=mx + c
        self.relu = nn.ReLU()
        #activation fucntion, to introduce non -linerity to the model sigmoid func
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        #second linear transformation
        #4layers - input-hidden, output-hidden 
        self.l3 = nn.Linear(hidden_size, num_classes)  
        #final linear transformation

    def forward(self, x):
        #defines how the NN will pass the input to forward layers, takes input tensor x(x is a multidimensional array)
        out = self.l1(x)
        #input tensor to fisrt layer
        out = self.relu(out)
        #applies activation func to the output of the first layer
        out = self.l2(out)
        #passes the output of the relu layer through the second linear layer
        out = self.relu(out)  
        #again activation
        out = self.l3(out)
        #passes the output of the second linear layer through the final layer
        return out
