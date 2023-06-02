import numpy as np
from .nn_layer import NNLayer


class MaxPool(NNLayer):

    def __init__(self, filter_dim, stride):
        super().__init__()
        self.name = 'max_pool'
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

    def repeat(self, Z, out_shape):
        
        Z_repeated = Z.repeat(self.filter_dim, axis=2).repeat(self.filter_dim, axis=3)
        Z_padded = np.zeros(out_shape)
        Z_padded[:, :, :Z_repeated.shape[2], :Z_repeated.shape[3]] += Z_repeated
        return Z_padded

    
    def forward(self, X):

        """
            Parameters:
                X: input to this layer ==> Shape : (m,c,h,w) where m = batch size
            
            Returns:
                Z: Maxpool(X) ==> Shape : (m, c, oh, ow 
                    where oh = (h-f)/s+1 , ow = (w-f)/s+1),
                    m=batch size, f=filter_dim, s=stride
                    
        """

        self.cache['X'] = X
        m, c, h, w = X.shape

        out_h = (h - self.filter_dim) // self.stride + 1
        out_w = (w - self.filter_dim) // self.stride + 1
        
        X_windows = self.get_sliding_windows(X, out_shape=(m, c, out_h, out_w), stride=self.stride)
        Z = np.max(X_windows, axis=(4, 5))
        
        if self.stride == self.filter_dim:
            Z_repeated = self.repeat(Z, X.shape)
            mask = np.equal(Z_repeated, X).astype(int)
            self.cache['mask'] = mask
        
        else:
            self.cache['Z'] = Z
        return Z

    def backward(self, dZ, lr):
        
        """
        Parameters:
            dZ: d(loss)/d(output from this layer) ==> Shape : (m, c, oh, ow)

        Returns:
            dX: d(loss) / d(input to this layer) ==> Shape : (m, c, h, w)
        
        """
        if self.filter_dim == self.stride:
            
            dZ_repeated = self.repeat(dZ, self.cache['X'].shape)
            dX = np.multiply(dZ_repeated, self.cache['mask'])
        
        else:
            dX = np.zeros(self.cache['X'].shape)
            out_h, out_w = self.cache['Z'].shape[2:]
            
            for i in range(out_h):
                for j in range(out_w):
                    
                    ii =  i * self.stride
                    jj = j * self.stride
                    
                    Z_window = self.cache['Z'][:, :, i, j][:, :, np.newaxis, np.newaxis]
                    dZ_window = dZ[:, :, i, j][:, :, np.newaxis, np.newaxis]
                    X_window = self.cache['X'][:, :, ii:ii+self.filter_dim, jj:jj+self.filter_dim]
                    
                    mask = np.equal(X_window, Z_window)
                    dX[:, :, ii:ii+self.filter_dim, jj:jj+self.filter_dim] += np.multiply(dZ_window, mask)   

        return dX
    
    
    def __str__(self):
        return "MaxPool f: {} s: {}".format(self.filter_dim, self.stride)

    def save_params(self):
        return {"name": self.name}
    
    def load_params(self, params):
        return