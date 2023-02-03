import numpy as np
from .nn_layer import NNLayer

class Convolution(NNLayer):

    def __init__(self, out_channels, filter_dim, stride=1, padding=0):
        super().__init__()
        
        self.out_channels = out_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding

        self.params = {
            'W': np.random.rand(self.filter_dim, self.filter_dim, self.)
        }

    def forward(self, X):

        """
            Convolution operation
                
            Parameters:
                X: input to this layer ==> Shape : (m, h, w, c) where m = batch size
            
            Returns:
                Z: conv(X, filters) ==> Shape : (m, (h+2p-f)/s+1, (w+2p-f)/s+1, oc) 
                    where p=padding, f=filter_dim, s=stride, oc=output_channes
        """

        return 

    def backward(self, dZ, lr):

        return 
    

    def __str__(self):
        return "Convolution out: {} f: {} s: {} p: {}".format(self.out_channels, self.filter_dim, self.stride, self.padding)

        