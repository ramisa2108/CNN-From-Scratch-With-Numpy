import numpy as np
from .nn_layer import NNLayer

class SoftMax(NNLayer):

    def __init__(self):
        
        super().__init__()    
        
        # Cache for storing result of forward pass. Will be needed for back-propagation
        self.cache = {'Z': None}

    def forward(self, X):

        """
            Softmax activation function: 
                Softmax(x_i) = e^(x_i) / sum(e^X)

            Parameters:
                X: input to this layer ==> Shape : (n, m) where m = batch size, n = 10 (number of labels)

            Returns:
                Z: Softmax(X) ==> Shape : (n, m) 
        """
        ex = np.exp(X)
        sum = np.sum(ex, axis=0)
        self.cache['Z'] = ex / sum
        
        return self.cache['Z']

    def backward(self, Y, lr):

        """
            Parameters:
                Y: ground truth ==> Shape : (n, m) where m = batch size, n = 10 (number of labels)

            Returns:
                dX: d(loss) / d(input to this layer) ==> Shape : (n, m)
        """
        return self.cache['Z'] - Y

    def __str__(self):
        return "Softmax"
