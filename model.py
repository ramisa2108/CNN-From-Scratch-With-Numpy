from layers import *
import pickle
import os
import numpy as np

class CNN:

    def __init__(self, model_desc, model_weight_file=None):

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

        print(self.model_layers)

        if model_weight_file is not None:
            self.load_model_weights(model_weight_file)

    def save_model_weights(self, save_dir, file_prefix):

        print('Saving model in {} as {}_model_ckpt.pkl'.format(save_dir, file_prefix))
        pickle.dump(self.model_layers, open(os.path.join(save_dir, file_prefix + '_model_ckpt.pkl', 'wb')))
        

    def load_model_weights(self, model_weight_file):

        with open(model_weight_file, "rb") as file:
            self.model_layers = pickle.load(file)

    def forward(self, X):
        
        for i, layer in enumerate(self.model_layers):
            X = layer.forward(X)
            
        return np.transpose(X)

    def backward(self, Y, lr):
        
        dY = np.transpose(Y)
        
        for layer in self.model_layers[::-1]:
            dY = layer.backward(dY, lr)
        return dY
        
            


            

    