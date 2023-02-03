import numpy as np
from .nn_layer import NNLayer

class MaxPool(NNLayer):

    def __init__(self, filter_dim, stride):
        super().__init__()
        self.filter_dim = filter_dim
        self.stride = stride
        self.cache = {'X': None}

    def forward(self, X):

        """
            Maxpool()
            Parameters:
                X: input to this layer ==> Shape : (m,h,w,c) where m = batch size
            
            Returns:
                Z: Maxpool(X) ==> Shape : (m, (h-f)/s+1, (w-f)/s+1, c) 
                                          where m=batch size, f=filter_dim, s=stride
        """

        return 

    def backward(self, dZ, lr):


        return 
        

    def __str__(self):
        return "MaxPool f: {} s: {}".format(self.filter_dim, self.stride)
