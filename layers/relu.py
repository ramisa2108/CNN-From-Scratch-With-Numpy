import numpy as np
from .nn_layer import NNLayer

class ReLU(NNLayer):

    def __init__(self):
        super().__init__()
       
        # Cache for storing input to forward pass. Will be needed for back-propagation
        self.cache = {'X': None}

    def forward(self, X):
        
        """
            Relu activation function: 
                ReLU(x) = max(x, 0)

            Parameters:
                X: input to this layer ==> Shape : (m, *) where m = batch size

            Returns:
                Z: RelU(X) ==> Shape : (m, *)
        """
        
        self.cache['X'] = X
        return X * (X > 0)

    def backward(self, dZ, lr):
        """
            Parameters:
                dZ: d(loss)/d(output of this layer) ==> Shape : (m, *) where m = batch size

            Returns:
                dX: d(loss) / d(input to this layer) ==> Shape : (m, *)
        """
        return dZ * (self.cache['X'] > 0)

    def __str__(self):
        return "ReLU"
