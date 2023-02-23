import os
from abc import ABC, abstractclassmethod
import numpy as np
import pandas as pd
import cv2
import pickle
from sklearn import metrics
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'

    train_image_folder = os.path.join(data_folder, 'train')
    test_image_folder = os.path.join(data_folder, 'test')
    train_labels = os.path.join(data_folder, 'train.csv')
    test_labels = os.path.join(data_folder, 'test.csv')

    train_size = None
    val_prop = 0.2
    test_size = None


    model_folder = 'saved_models'
    model_weight_file = os.path.join(model_folder, 'big_model_weights.pickle')
    load_pretrained = False

    output_folder = 'outputs'

    NUM_LABELS = 10
    READ_COLORED_IMAGE = 0
    LEARNING_RATE = 0.01
    IMAGE_DIMS = (32, 32)
    BATCH_SIZE = 32
    EPOCHS = 50

conf = Config()


def one_hot_encoding(Y):

    m = Y.shape[0]
    one_hot = np.zeros((m, conf.NUM_LABELS))
    one_hot[np.arange(m), Y] = 1
    return one_hot

def get_labels(y):
    return np.argmax(y, axis=1)


def split_train_set(X, y, val_prop):

    total_size = X.shape[0]
    train_size = total_size - int(total_size * val_prop)

    ids = np.arange(total_size)
    np.random.shuffle(ids)

    train_X, train_y = X[ids[:train_size]], y[ids[:train_size]]
    val_X, val_y = X[ids[train_size:]], y[ids[train_size:]]
    
    return (train_X, train_y), (val_X, val_y)

def load_datasets(image_folder, label_file, dataset_size=None):

    labels = pd.read_csv(label_file)
    map = dict(zip(labels['filename'], labels['digit']))

    all_images = [l for l in os.listdir(image_folder) if l.endswith('.png')]

    if dataset_size is not None:
        np.random.shuffle(all_images)
        all_images = all_images[:dataset_size]

    X = []
    y = []
    

    for image_file in all_images:
        img = cv2.imread(os.path.join(image_folder, image_file), conf.READ_COLORED_IMAGE)
        img = cv2.resize(img, conf.IMAGE_DIMS)
        img = (255.0 - img) / 255.0

        X.append(img)
        y.append(map[image_file])
        
    X = np.array(X)
    y = np.array(y)


    # for RGB
    if conf.READ_COLORED_IMAGE:
        X = X.transpose(0, 3, 1, 2)
    # for Grayscale
    else:
        X = X[:, np.newaxis, :, :]
    
    list_name = 'train_file_list.pickle'
    pickle.dump(all_images, open(os.path.join(conf.output_folder, list_name), "wb"))
    
    print("{} images loaded from {}, labels loaded {}".format(len(X), image_folder, len(y)))
    return X, y


def load_model_description(file_name):

    with open(file_name, 'rb') as file:
        layers = pickle.load(file)

    return layers

def load_model_weights(file_name):
    
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'rb') as file:
        weights = pickle.load(file)
    return weights

def cross_entropy_loss(true_labels, output_propabilities):
    return np.sum(-1.0 * true_labels * np.log(output_propabilities))


def accuracy(true_labels, predicted_labels):
    return metrics.accuracy_score(true_labels, predicted_labels)


def macro_f1_score(true_labels, predicted_labels):

    return metrics.f1_score(true_labels, predicted_labels, average='macro')

def label_wise_score(true_labels, predicted_labels):

    print("Accuracy and F1 score for labels:")
    for l in range(conf.NUM_LABELS):
        pos = np.where(true_labels == l)
        f1 = metrics.f1_score(true_labels[pos], predicted_labels[pos], average='macro')
        acc = metrics.accuracy_score(true_labels[pos], predicted_labels[pos])
        print("{} acc: {}, f1: {}".format(l, acc, f1))


class NNLayer(ABC):


    @abstractclassmethod
    def forward(self, X):
        pass

    @abstractclassmethod
    def backward(self, dZ, lr):
        pass

    @abstractclassmethod
    def load_params(self, params):
        pass

    @abstractclassmethod
    def save_params(self):
        pass

    
    def __str__(self):
        return "NNLayer"

class Convolution(NNLayer):

    def __init__(self, out_channels, filter_dim, stride=1, padding=0):
        super().__init__()

        self.name = 'conv'
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
        self.params['b'] = np.zeros((self.out_channels, 1, 1))

        self.gradients['dW'] = np.zeros_like(self.params['W'])
        self.gradients['db'] = np.zeros_like(self.params['b'])

    

    def get_sliding_windows(self, X, out_shape, padding, stride):

        padded_X = X.copy()

        if padding > 0:
            padded_X = np.pad(padded_X, pad_width=((0,), (0,), (padding,), (padding,)))
        elif padding < 0:
            padded_X = X[:, :, -padding: padding, -padding: padding]

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
        if self.params['W'] is None:
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

        

        dZ_windows = self.get_sliding_windows(dZ_dilated, out_shape=dZ.shape[:2] + self.cache['X'].shape[2:], padding=padding, stride=1)
        W_180 = np.rot90(self.params['W'], k=2, axes=(2,3))

        dX = np.einsum('mohwij, ocij -> mchw', dZ_windows, W_180)
        
        self.gradients['dW'] = 1/m * np.einsum('mchwij, mohw -> ocij', self.cache['X_windows'], dZ)
        self.gradients['db'] = 1/m * np.einsum('ijkl->j', dZ)[:,np.newaxis, np.newaxis]
        self.update_params(lr)
        return dX
    

    def update_params(self, lr):

        self.params['W'] = self.params['W'] - lr * self.gradients['dW']   # W = W - alpha * dW
        self.params['b'] = self.params['b'] - lr * self.gradients['db']   # b = b - alpha * db

    

    def __str__(self):
        return "Convolution out: {} f: {} s: {} p: {}".format(self.out_channels, self.filter_dim, self.stride, self.padding)

    
    def save_params(self):

        params = {}
        params['name'] = self.name
        params['W'] = self.params['W']
        params['b'] = self.params['b']
        params['dW'] = self.gradients['dW']
        params['db'] = self.gradients['db']
        params['in_channels'] = self.in_channels       
        return params
        

    def load_params(self, params):
        
        self.in_channels = params['in_channels']
        self.params['W'] = params['W']
        self.params['b'] = params['b']
        self.gradients['dW'] = np.zeros_like(params['W'])
        self.gradients['db'] = np.zeros_like(params['b'])

        assert (self.params['W'].shape == (self.out_channels, self.in_channels, self.filter_dim, self.filter_dim))
        assert (self.params['b'].shape == (self.out_channels, 1, 1))

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
    
class FullyConnected(NNLayer):

    def __init__(self, out_dim):

        super().__init__()

        self.name = 'fc'
        
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

    def save_params(self):

        params = {}
        params['name'] = self.name
        params['W'] = self.params['W']
        params['b'] = self.params['b']
        params['dW'] = self.gradients['dW']
        params['db'] = self.gradients['db']
        params['in_dim'] = self.in_dim       
        return params
        

    def load_params(self, params):
        
        self.in_dim = params['in_dim']
        self.params['W'] = params['W']
        self.params['b'] = params['b']
        self.gradients['dW'] = np.zeros_like(params['W'])
        self.gradients['db'] = np.zeros_like(params['b'])

        assert(self.params['W'].shape == (self.out_dim, self.in_dim))
        assert(self.params['b'].shape == (self.out_dim, 1))

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
    
class ReLU(NNLayer):

    def __init__(self):
        super().__init__()
        self.name = 'relu'
       
        # Cache for storing input to forward pass. Will be needed for back-propagation
        self.cache = {'X': None}

    def forward(self, X):
        
        """
            Relu activation function: 
                ReLU(x) = max(x, 0)

            Parameters:
                X: input to this layer ==> Shape : (m, *) where m = batch size

            Returns:
                Z: RelU(X) ==> Shape : (m, *)
        """
        
        self.cache['X'] = X
        return X * (X > 0)

    def backward(self, dZ, lr):
        """
            Parameters:
                dZ: d(loss)/d(output of this layer) ==> Shape : (m, *) where m = batch size

            Returns:
                dX: d(loss) / d(input to this layer) ==> Shape : (m, *)
        """
        return dZ * (self.cache['X'] > 0)

    def __str__(self):
        return "ReLU"


    def save_params(self):
        return {"name": self.name}
    
    def load_params(self, params):
        return

class SoftMax(NNLayer):

    def __init__(self):
        
        super().__init__()    
        self.name = 'softmax'
        
        # Cache for storing result of forward pass. Will be needed for back-propagation
        self.cache = {'Z': None}

    def forward(self, X):

        """
            Softmax activation function: 
                Softmax(x_i) = e^(x_i) / sum(e^X)

            Parameters:
                X: input to this layer ==> Shape : (n, m) where m = batch size, n = 10 (number of labels)

            Returns:
                Z: Softmax(X) ==> Shape : (n, m) 
        """
        ex = np.exp(X)
        sum = np.sum(ex, axis=0)
        self.cache['Z'] = ex / sum
        
        return self.cache['Z']

    def backward(self, Y, lr):

        """
            Parameters:
                Y: ground truth ==> Shape : (n, m) where m = batch size, n = 10 (number of labels)

            Returns:
                dX: d(loss) / d(input to this layer) ==> Shape : (n, m)
        """
        return self.cache['Z'] - Y

    def __str__(self):
        return "Softmax"


    def save_params(self):
        return {"name": self.name}
    
    def load_params(self, params):
        return

class CNN:

    def __init__(self, model_weight=None):

        
        conv1 = Convolution(out_channels=8, filter_dim=5, stride=1, padding=0)
        relu1 = ReLU()
        max_pool1 = MaxPool(filter_dim=3, stride=3)

        conv2 = Convolution(out_channels=16, filter_dim=3, stride=1, padding=0)
        relu2 = ReLU()
        max_pool2 = MaxPool(filter_dim=3, stride=3)

        flatten1 = Flattening() 
        fc1 = FullyConnected(out_dim=200)
        fc2 = FullyConnected(out_dim=10)
        softmax1 = SoftMax()

        self.model_layers = [conv1, relu1, max_pool1, conv2, relu2, max_pool2, flatten1, fc1, fc2, softmax1]        
        
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

import numpy as np
from config import Config
from utils import *
from model import CNN
from tqdm import tqdm
import time
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


conf = Config()


def calculate_metrics(y_true, y_predicted_prob, label_wise=False):

    
    y_true_labels = get_labels(y_true)
    y_predicted_labels = get_labels(y_predicted_prob)

    ce_loss = cross_entropy_loss(y_true, y_predicted_prob) / len(y_true)
    f1_score = macro_f1_score(y_true_labels, y_predicted_labels)
    acc = accuracy(y_true_labels, y_predicted_labels)

    if label_wise:
        label_wise_score(y_true_labels, y_predicted_labels)
    
    return (ce_loss, f1_score, acc)


def eval_epoch(model, val_X):
    
    n_val_batches = (val_X.shape[0] + conf.BATCH_SIZE - 1) // conf.BATCH_SIZE

    val_pred = np.zeros([0,10])

    for i in tqdm(range(n_val_batches)):
        val_X_batch = val_X[i * conf.BATCH_SIZE : (i+1) * conf.BATCH_SIZE]
        y_out_prob = model.predict(val_X_batch)
        val_pred = np.vstack([val_pred, y_out_prob])
    
    return val_pred


def train_epoch(model, train_X, train_y):

    
    train_loss = 0.0
    n_train_batches = (train_X.shape[0] + conf.BATCH_SIZE - 1) // conf.BATCH_SIZE

    for i in tqdm(range(n_train_batches)):
        train_X_batch = train_X[i * conf.BATCH_SIZE : (i+1) * conf.BATCH_SIZE]
        train_y_batch = train_y[i * conf.BATCH_SIZE : (i+1) * conf.BATCH_SIZE]
    
        model.train_model(train_X_batch, train_y_batch, conf.LEARNING_RATE)
        y_out_prob = model.predict(train_X_batch)
        train_loss += cross_entropy_loss(train_y_batch, y_out_prob)


    train_loss /= train_X.shape[0]
        
    return train_loss

def plot_metrics(x, y, x_label, y_label):
    
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + ' vs. ' + y_label)
    plt.savefig(os.path.join(conf.output_folder, x_label + ' vs ' + y_label + '.png'))
    plt.clf()
        

if __name__ == '__main__':

    np.random.seed(0)
    np.set_printoptions(precision=2)

    if not os.path.exists(conf.model_folder):
        os.mkdir(conf.model_folder)
    if not os.path.exists(conf.output_folder):
        os.mkdir(conf.output_folder)

    train_X, train_y_labels = load_datasets(conf.train_image_folder, conf.train_labels, conf.train_size)
    train_y = one_hot_encoding(train_y_labels)

    (train_X, train_y), (val_X, val_y) = split_train_set(train_X, train_y, conf.val_prop)

    
    model_weights = None
    if conf.load_pretrained:
        model_weights = load_model_weights(conf.model_weight_file)

    model = CNN(model_weights)

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    best_val_f1_score = 0.0

    for epoch in range(conf.EPOCHS):
        
        t1 = time.time()
        train_loss = train_epoch(model, train_X, train_y)

        val_pred = eval_epoch(model, val_X)
        val_loss, val_f1_score, val_accuracy = calculate_metrics(val_y, val_pred)
        
        t2 = time.time()
        
        print('Epoch {}: val_acc: {}  val_F1: {} val_loss: {} train_loss: {} time: {}'.format(epoch, val_accuracy, val_f1_score, val_loss, train_loss,t2-t1))

        if val_f1_score > best_val_f1_score:
            best_val_f1_score = val_f1_score
            model.save_model_weights(conf.model_weight_file)

        train_losses += [train_loss]
        val_losses += [val_loss]
        val_accs += [val_accuracy]
        val_f1s += [val_f1_score]

    print('Training completed. Testing...')

    plot_metrics(x=np.arange(len(train_losses)), y=train_losses, x_label='Epoch', y_label='Training Loss')
    plot_metrics(x=np.arange(len(val_losses)), y=val_losses, x_label='Epoch', y_label='Validation Loss')
    plot_metrics(x=np.arange(len(val_accs)), y=val_accs, x_label='Epoch', y_label='Validation Accuracy')
    plot_metrics(x=np.arange(len(val_f1s)), y=val_f1s, x_label='Epoch', y_label='Validation Macro F1')


    model_weights = load_model_weights(conf.model_weight_file)
    model = CNN(model_weights)

    val_pred = eval_epoch(model, val_X)
    val_loss, val_f1_score, val_accuracy = calculate_metrics(val_y, val_pred)

    print('Best Val acc: {}, loss: {}, F1: {}'.format(val_accuracy, val_loss, val_f1_score))    

    val_y_labels = get_labels(val_y)
    val_pred_labels = get_labels(val_pred)
    
    cm = confusion_matrix(val_y_labels, val_pred_labels)
    sns.heatmap(cm, annot=True)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title('Confusion Matrix For Validation Set')
    plt.savefig(os.path.join(conf.output_folder, 'Confusion Matrix For Validation Set'))
    plt.clf()