from layers import *
import pickle
import numpy as np
from layers import *

class CNN:

    def __init__(self, model_weight=None):

        conv1 = convolution.Convolution(out_channels=8, filter_dim=5, stride=1, padding=0)
        relu1 = relu.ReLU()
        max_pool1 = max_pool.MaxPool(filter_dim=3, stride=3)

        conv2 = convolution.Convolution(out_channels=16, filter_dim=3, stride=1, padding=0)
        relu2 = relu.ReLU()
        max_pool2 = max_pool.MaxPool(filter_dim=3, stride=3)

        flatten1 = flattening.Flattening() 
        fc1 = fully_connected.FullyConnected(out_dim=200)
        relu3 = relu.ReLU()
        fc2 = fully_connected.FullyConnected(out_dim=10)
        relu4 = relu.ReLU()
        softmax1 = softmax.SoftMax()

        self.model_layers = [conv1, relu1, max_pool1, conv2, relu2, max_pool2, flatten1, fc1, relu3, fc2, relu4, softmax1]        
        
        if model_weight is not None:
            self.load_model_weights(model_weight)

    def save_model_weights(self, file_path):

        print('Saving model in {}'.format(file_path))
        layer_params = []
        for layer in self.model_layers:
            layer_params += [layer.save_params()]

        pickle.dump(layer_params, open(file_path, 'wb'))
        

    def load_model_weights(self, model_weights):

        if len(model_weights) != len(self.model_layers):
            raise('Layers dont match')

        for i, layer in enumerate(self.model_layers):
            params = model_weights[i]
            if params['name'] != layer.name:
                raise('Layers dont match {} and {}'.format(params['name'], layer.name))

            layer.load_params(params)
        
    def forward(self, X):
        
        for layer in self.model_layers:
            X = layer.forward(X)
            
        return np.transpose(X)


    def backward(self, Y, lr):
        
        dY = np.transpose(Y)
        
        for layer in self.model_layers[::-1]:
            dY = layer.backward(dY, lr)
        return
    
    def train_model(self, X, y, lr):

        for layer in self.model_layers:
            X = layer.forward(X)
        
        dy = np.transpose(y)

        for layer in self.model_layers[::-1]:
            dy = layer.backward(dy, lr)
         
        
    def predict(self, X):

        for layer in self.model_layers:
            X = layer.forward(X)
        
        y_probs = np.transpose(X)
        return y_probs