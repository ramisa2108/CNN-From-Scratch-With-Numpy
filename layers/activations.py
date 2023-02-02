import numpy as np
from abc import ABC, abstractclassmethod


class Activation(ABC):

    def __call__(self, X):
        return self.forward(X)

    @abstractclassmethod
    def forward(self, X):
        pass

    @abstractclassmethod
    def backward(self, dY):
        pass


class ReLU(Activation):

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


class SoftMax(Activation):

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

