import numpy as np
from nn_layer import NNLayer

class ReLU(NNLayer):

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, X: np.ndarray):
        
        """
            Relu activation function: 
                ReLU(x) = max(x, 0)

            Parameters:
                X: input to this layer. 

            Returns:
                RelU(X)
        """
        
        self.cache['X'] = X
        return X * (X > 0)

    def backward(self, dY: np.ndarray):
        """
            Parameters:
                dY: d(loss)/d(output of this layer)

            Returns:
                d(loss) / d(input to this layer)
        """
        return dY * (self.cache['X'] > 0)
