import numpy as np
from nn_layer import NNLayer

class SoftMax(NNLayer):

    def __init__(self):
        super().__init__()    
        self.cache = None

    def forward(self, X):

        """
            Softmax activation function: 
                Softmax(x_i) = e^(x_i) / sum(e^X)

            Parameters:
                X: input to this layer. shape : (m x n) where m = batch size

            Returns:
                Z: Softmax(X)
        """
        ex = np.exp(X)
        sum = np.sum(ex, axis=1).reshape([-1, 1])
        self.cache['Z'] = ex / sum
        return self.cache['Z']

    def backward(self, Y):

        """
            Parameters:
                Y: ground truth

            Returns:
                dX: d(loss) / d(input to this layer)
        """
        return self.cache['Z'] - Y 

