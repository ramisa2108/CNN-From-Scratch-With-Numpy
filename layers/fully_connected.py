import numpy as np
from .nn_layer import NNLayer

class FullyConnected(NNLayer):

    def __init__(self, out_dim):

        super().__init__()
        
        self.out_dim = out_dim
        self.in_dim = None

        self.params = {'W': None, 'b': None}
        self.gradients = {'dW': None, 'db': None}

        # Cache for storing input to forward pass. Will be needed for back-propagation
        self.cache = {'X': None}

    
    def xavier_init(self, in_dim):
        
        self.in_dim = in_dim
        
        # self.params['W'] = np.random.randn(self.out_dim, self.in_dim) * np.sqrt(1.0 / self.in_dim)
        # self.params['b'] = np.random.randn(1, self.out_dim) * np.sqrt(1.0 / self.out_dim) 
        
        self.params['W'] = np.random.randn(self.out_dim, self.in_dim) * np.sqrt(2.0 / self.in_dim)
        # self.params['b'] = np.random.randn(self.out_dim, 1) * np.sqrt(1.0 / self.out_dim) 
        self.params['b'] = np.zeros((self.out_dim, 1)) 
        
        self.gradients['dW'] = np.zeros_like(self.params['W'])
        self.gradients['db'] = np.zeros_like(self.params['b'])

        

    def forward(self, X):

        """
            Forward propagation between 2 dense layers
                FC(X) = W * X + b
            
            Parameters:
                X: input to this layer ==> Shape : (in_dim, m) where m = batch size
            
            Returns:
                Z: FC(X) ==> Shape : (out_dim, m)

        """
        
        # set the parameters in the first pass through forward prop
        if self.in_dim is None:
            self.xavier_init(X.shape[0])
        
        self.cache['X'] = X
        # Z =  self.params['W'] @ X + self.params['b']
        Z = np.einsum('ij, jk -> ik', self.params['W'], X) + self.params['b'] # Z = W * X + b
        return Z

    def backward(self, dZ, lr):

        """
        Parameters:
            dZ: d(loss)/d(output of this layer) ==> Shape : (out_dim, m) where m = batch size

        Returns:
            dX: d(loss) / d(input to this layer) ==> Shape : (in_dim, m)
            Also calculates dW and db
        """
        
        m = self.cache['X'].shape[1]
        
        # self.gradients['dW'] = 1.0/m * dZ @ np.transpose(self.cache['X'])
        self.gradients['dW'] = 1.0/m * np.einsum('ij, kj -> ik', dZ, self.cache['X'])
        # self.gradients['db'] = 1.0/m * np.sum(dZ, axis=1)  #.reshape(self.params['b'].shape)
        self.gradients['db'] = 1.0/m * np.einsum('kij->ik', [dZ])
        
        # dX = np.transpose(self.params['W']) @ dZ
        dX = np.einsum('ji, jk->ik', self.params['W'], dZ)
        self.update_params(lr)

        return dX
    
    def update_params(self, lr):

        
        self.params['W'] = self.params['W'] - lr * self.gradients['dW']   # W = W - alpha * dW
        self.params['b'] = self.params['b'] - lr * self.gradients['db']   # b = b - alpha * db
        

    def __str__(self):
        return "Fully Connected out: {}".format(self.out_dim)
