import numpy as np
from .nn_layer import NNLayer



class Flattening(NNLayer):

    def __init__(self):
        super().__init__()
        self.name = 'flatten'
        self.cache = {'in_shape': None}
    
    def forward(self, X):
        """
            Converts each data point from nD matrix to a 1D vector

            Parameter:
                X: input to this layer ==> Shape : (m, (h,w,c)) where m = batch size
            
            Returns:
                Z: X flattened along rows ==> Shape : (h*w*c, m)
                making the features column wise for the fully connected layers
        """
        self.cache['in_shape'] = X.shape
        Z = X.reshape((X.shape[0], np.prod(X.shape[1:])))
        Z = np.transpose(Z)
        return Z

    def backward(self, dZ, lr):

        """
            Parameter:
                dZ: d(loss)/d(output of this layer) ==> Shape : (h*w*c, m) where m = batch size
            Returns:
                dX : dZ transposed and reshaped to match the input to this layer
        """
        # return  dZ.reshape(self.cache['in_shape'])
        dX = np.transpose(dZ)
        return dX.reshape(self.cache['in_shape'])
    
    def __str__(self):
        return "Flattening"

    def save_params(self):
        return {"name": self.name}
    
    def load_params(self, params):
        return