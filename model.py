from layers import *
import pickle
import os
import numpy as np

class CNN:

    def __init__(self, model_desc, model_weight=None):

        self.model_layers = []
        for l in model_desc:
            if l['name'] == 'conv':
                layer = convolution.Convolution(l['out_channels'], l['filter_dim'], l['stride'], l['padding'])
            elif l['name'] == 'flatten':
                layer = flattening.Flattening()
            elif l['name'] == 'fc':
                layer = fully_connected.FullyConnected(l['out_dim'])
            elif l['name'] == 'max_pool':
                layer = max_pool.MaxPool(l['filter_dim'], l['stride'])
            elif l['name'] == 'relu':
                layer = relu.ReLU()
            elif l['name'] == 'softmax':
                layer = softmax.SoftMax()
            else:
                print('Unknown layer')
            
            self.model_layers.append(layer)

        
        if model_weight is not None:
            self.load_model_weights(model_weight)

    def save_model_weights(self, save_dir, file_prefix):

        print('Saving model in {} as {}_model_weights.pkl'.format(save_dir, file_prefix))
        layer_params = []
        for layer in self.model_layers:
            layer_params += [layer.save_params()]

        pickle.dump(layer_params, open(os.path.join(save_dir, file_prefix + '_model_weights.pkl'), 'wb'))
        

    def load_model_weights(self, model_weights):

        if len(model_weights) != self.model_layers:
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


            

    