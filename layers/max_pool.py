import numpy as np
from .nn_layer import NNLayer

class MaxPool(NNLayer):

    def __init__(self, filter_dim, stride):
        super().__init__()
        self.filter_dim = filter_dim
        self.stride = stride
        self.cache = {'X': None}



    def get_sliding_windows(self, X, out_shape, stride):

        
        out_m, out_c, out_h, out_w = out_shape
        stride_m, stride_c, stride_h, stride_w = X.strides

        return np.lib.stride_tricks.as_strided(
            X, (out_m, out_c, out_h, out_w, self.filter_dim, self.filter_dim),
            (stride_m, stride_c, stride_h * stride, stride_w * stride, stride_h, stride_w)
        )

    
    
    def forward(self, X):

        """
            Parameters:
                X: input to this layer ==> Shape : (m,c,h,w) where m = batch size
            
            Returns:
                Z: Maxpool(X) ==> Shape : (m, c, oh, ow 
                    where m=batch size, f=filter_dim, s=stride
                    oh = (h-f)/s+1 , ow = (w-f)/s+1)
        """
        self.cache['X'] = X
        m, c, h, w = X.shape

        out_h = (h - self.filter_dim) // self.stride + 1
        out_w = (w - self.filter_dim) // self.stride + 1
        

        X_windows = self.get_sliding_windows(Xout_shape=(m, c, out_h, out_w), stride=self.stride)
        self.cache['X_windows'] = X_windows

        Z = np.max(X_windows, axis=(4, 5))
        return Z

    def backward(self, dZ, lr):
        

        return 
        

    def __str__(self):
        return "MaxPool f: {} s: {}".format(self.filter_dim, self.stride)
