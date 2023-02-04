import numpy as np
from .nn_layer import NNLayer

class Convolution(NNLayer):

    def __init__(self, out_channels, filter_dim, stride=1, padding=0):
        super().__init__()
        
        self.in_channels = None
        self.out_channels = out_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding

        self.params = {'W': None, 'b': None}
        self.gradients = {'dW': None, 'db': None}

        self.cache = {'X': None, 'X_windows': None}


    def xavier_init(self, in_channels):
        
        self.in_channels = in_channels
        self.params['W'] = np.random.rand(self.out_channels, in_channels, self.filter_dim, self.filter_dim) * np.sqrt(2.0 / (self.filter_dim * self.filter_dim * in_channels))
        self.params['b'] = np.zeros(self.out_channels, 1, 1)

        self.gradients['dW'] = np.zeros_like(self.params['W'])
        self.gradients['db'] = np.zeros_like(self.params['b'])

    

    def get_sliding_windows(self, X, out_shape, padding, stride):

        padded_X = X.copy()

        if padding != 0:
            padded_X = np.pad(padded_X, pad_width=((0,), (0,), (padding,), (padding,)))

        out_m, out_c, out_h, out_w = out_shape
        stride_m, stride_c, stride_h, stride_w = padded_X.strides

        return np.lib.stride_tricks.as_strided(
            padded_X, (out_m, out_c, out_h, out_w, self.filter_dim, self.filter_dim),
            (stride_m, stride_c, stride_h * stride, stride_w * stride, stride_h, stride_w)
        )

    def forward(self, X):

        """
            Convolution operation
                
            Parameters:
                X: input to this layer ==> Shape : (m, c, h, w) 
                where m = batch size, c = input_channels, h = height, w = width
            
            Returns:
                Z: conv(X, filters) ==> Shape : (m, oc, oh, ow) 
                    where p=padding, f=filter_dim, s=stride,
                    oh = (h+2p-f)/s+1 , ow = (w+2p-f)/s+1) , oc=output_channels 
        """
        if self.in_channels is None:
            self.xavier_init(X.shape[1])
        
        self.cache['X'] = X
        m, c, h, w = X.shape

        out_h = (h + 2 * self.padding - self.filter_dim) // self.stride + 1
        out_w = (w + 2 * self.padding - self.filter_dim) // self.stride + 1
        

        X_windows = self.get_sliding_windows(X, out_shape=(m, c, out_h, out_w), padding=self.padding, stride=self.stride)
        self.cache['X_windows'] = X_windows

        Z = np.einsum('mchwij, ocij -> mohw', X_windows, self.params['W']) + self.params['b']
        return Z

    def backward(self, dZ, lr):

        """
        Parameters:
            dZ: d(loss)/d(output of this layer) ==> Shape : (m, oc, oh, ow)

        Returns:
            dX: d(loss) / d(input to this layer) ==> Shape : (m, c, h, w)
            Also calculates dW and db
        """
        m = self.cache['X'].shape[0]
        
        padding = self.filter_dim - 1
        if self.padding:
            padding -= self.padding

        dilate = self.stride-1
        dZ_dilated = dZ.copy()
        dZ_dilated = np.insert(dZ_dilated, dilate * list(range(1, dZ.shape[2])), 0, axis=2)
        dZ_dilated = np.insert(dZ_dilated, dilate * list(range(1, dZ.shape[3])), 0, axis=3)

        dZ_windows = self.get_sliding_windows(dZ_dilated, out_shape=self.cache['X'].shape, padding=padding, stride=1)
        W_180 = np.rot90(self.params['W'], k=2, axes=(2,3))

        
        self.gradients['dW'] = 1/m * np.einsum('mchwij, mohw -> ocij', self.cache['X_windows'], dZ_dilated)
        dX = np.einsum('mohwij, ocij -> mchw', dZ_windows, W_180)
        self.gradients['db'] = 1/m * np.einsum('ijkl->j', dZ)[:,np.newaxis, np.newaxis]
        self.update_params(lr)
        return dX
    

    def update_params(self, lr):

        self.params['W'] = self.params['W'] - lr * self.gradients['dW']   # W = W - alpha * dW
        self.params['b'] = self.params['b'] - lr * self.gradients['db']   # b = b - alpha * db


    def __str__(self):
        return "Convolution out: {} f: {} s: {} p: {}".format(self.out_channels, self.filter_dim, self.stride, self.padding)

        