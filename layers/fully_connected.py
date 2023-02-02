import numpy as np
from nn_layer import NNLayer

class FullyConnected(NNLayer):

    def __init__(self, out_dim):
        
        self.out_dim = out_dim
        self.in_dim = None
        self.weight = None
        self.bias = None
        self.cache = None

    
    def init_parameters(self, in_dim):
        # Xavier initialization
        self.in_dim = in_dim



    def forward(self, X):

        
        
        

        return 

    def backward(self, input, output):

        return 
        