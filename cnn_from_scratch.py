# Convolutional Neural Network from Scratch

# Import Statements
from sklearn.model_selection import train_test_split

import abc
import numpy as np
import math
import os
import csv
from utils import *
from config import Config

conf = Config()

# Model Component Definition
class ModelComponent(abc.ABC):
    @abc.abstractmethod
    def forward(self, u):
        pass
    
    @abc.abstractmethod
    def backward(self, del_v, lr):
        pass
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        pass
    
    def save_learnable_parameters(self):
        pass
    
    def set_learnable_parameters(self):
        pass

# Convolution Layer Definition
class ConvolutionLayer(ModelComponent):
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u_pad = None
    
    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'
    
    def forward(self, u):
        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.kernel_size + 2 * self.padding) / self.stride) + 1
        num_channels = u.shape[3]
        
        if self.weights is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            # ref: https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
            self.weights = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size, num_channels) * math.sqrt(2 / (self.kernel_size * self.kernel_size * num_channels))
        if self.biases is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.biases = np.zeros(self.num_filters)
        
        self.u_pad = np.pad(u, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        v = np.zeros((num_samples, output_dim, output_dim, self.num_filters))
        
        for k in range(num_samples):
            for l in range(self.num_filters):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.sum(self.u_pad[k, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, :] * self.weights[l]) + self.biases[l]
        
        return v
    
    def backward(self, del_v, lr):
        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        input_dim_pad = (input_dim - 1) * self.stride + 1
        output_dim = self.u_pad.shape[1] - 2 * self.padding
        num_channels = self.u_pad.shape[3]
        
        del_b = np.sum(del_v, axis=(0, 1, 2)) / num_samples
        del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, self.num_filters))
        del_v_sparse[:, :: self.stride, :: self.stride, :] = del_v
        weights_prime = np.rot90(np.transpose(self.weights, (3, 1, 2, 0)), 2, axes=(1, 2))
        del_w = np.zeros((self.num_filters, self.kernel_size, self.kernel_size, num_channels))
        
        for l in range(self.num_filters):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    del_w[l, i, j, :] = np.mean(np.sum(self.u_pad[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)
        
        del_u = np.zeros((num_samples, output_dim, output_dim, num_channels))
        del_v_sparse_pad = np.pad(del_v_sparse, ((0,), (self.kernel_size - 1 - self.padding,), (self.kernel_size - 1 - self.padding,), (0,)), mode='constant')
        
        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        del_u[k, i, j, l] = np.sum(del_v_sparse_pad[k, i: i + self.kernel_size, j: j + self.kernel_size, :] * weights_prime[l])
        
        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b
    
    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)
    
    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)

# Activation Layer Definition
class ActivationLayer(ModelComponent):
    def __init__(self):
        self.u = None
    
    def __str__(self):
        return 'ReLU'
    
    def forward(self, u):
        self.u = u
        v = np.copy(u)
        v[v < 0] = 0  # applying ReLU activation function
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(self.u)
        del_u[del_u > 0] = 1  # applying sign(x) function for x > 0
        del_u[del_u < 0] = 0  # applying sign(x) function for x < 0
        del_u = del_v * del_u
        return del_u

# Max Pooling Layer Definition
class MaxPoolingLayer(ModelComponent):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.u_shape = None
        self.v_map = None
    
    def __str__(self):
        return f'MaxPool(kernel={self.kernel_size}, stride={self.stride})'
    
    def forward(self, u):
        self.u_shape = u.shape
        
        num_samples = u.shape[0]
        input_dim = u.shape[1]
        output_dim = math.floor((input_dim - self.kernel_size) / self.stride) + 1
        num_channels = u.shape[3]
        
        v = np.zeros((num_samples, output_dim, output_dim, num_channels))
        self.v_map = np.zeros((num_samples, output_dim, output_dim, num_channels)).astype(np.int32)
        
        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        v[k, i, j, l] = np.max(u[k, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, l])
                        self.v_map[k, i, j, l] = np.argmax(u[k, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size, l])
        
        return v
    
    def backward(self, del_v, lr):
        del_u = np.zeros(self.u_shape)
        
        num_samples = del_v.shape[0]
        input_dim = del_v.shape[1]
        num_channels = del_v.shape[3]
        
        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(input_dim):
                    for j in range(input_dim):
                        position = tuple(sum(pos) for pos in zip((self.v_map[k, i, j, l] // self.kernel_size, self.v_map[k, i, j, l] % self.kernel_size), (i * self.stride, j * self.stride)))
                        del_u[(k,) + position + (l,)] = del_u[(k,) + position + (l,)] + del_v[k, i, j, l]
        
        return del_u

# Flattening Layer Definition
class FlatteningLayer(ModelComponent):
    def __init__(self):
        self.u_shape = None
    
    def __str__(self):
        return 'Flatten'
    
    def forward(self, u):
        self.u_shape = u.shape
        v = np.copy(u)
        v = np.reshape(v, (v.shape[0], np.prod(v.shape[1:])))
        v = np.transpose(v)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        del_u = np.transpose(del_u)
        del_u = np.reshape(del_u, self.u_shape)
        return del_u

# Fully Connected Layer Definition
class FullyConnectedLayer(ModelComponent):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.biases = None
        self.weights_matrix = None
        self.biases_vector = None
        self.u = None
    
    def __str__(self):
        return f'FullyConnected(output_dim={self.output_dim})'
    
    def forward(self, u):
        self.u = u
        
        if self.weights is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.weights = np.random.randn(self.output_dim, u.shape[0]) * math.sqrt(2 / u.shape[0])
        if self.biases is None:
            # ref: https://cs231n.github.io/neural-networks-2/#init
            self.biases = np.zeros((self.output_dim, 1))
        
        v = self.weights @ u + self.biases
        return v
    
    def backward(self, del_v, lr):
        del_w = (del_v @ np.transpose(self.u)) / del_v.shape[1]
        del_b = np.reshape(np.mean(del_v, axis=1), (del_v.shape[0], 1))
        del_u = np.transpose(self.weights) @ del_v
        self.update_learnable_parameters(del_w, del_b, lr)
        return del_u
    
    def update_learnable_parameters(self, del_w, del_b, lr):
        self.weights = self.weights - lr * del_w
        self.biases = self.biases - lr * del_b
    
    def save_learnable_parameters(self):
        self.weights_matrix = np.copy(self.weights)
        self.biases_vector = np.copy(self.biases)
    
    def set_learnable_parameters(self):
        self.weights = self.weights if self.weights_matrix is None else np.copy(self.weights_matrix)
        self.biases = self.biases if self.biases_vector is None else np.copy(self.biases_vector)

# Softmax Layer Definition
class SoftmaxLayer(ModelComponent):
    def __init__(self):
        pass
    
    def __str__(self):
        return 'Softmax'
    
    def forward(self, u):
        v = np.exp(u)
        v = v / np.sum(v, axis=0)
        return v
    
    def backward(self, del_v, lr):
        del_u = np.copy(del_v)
        return del_u

# Model Definition
class Model:
    def __init__(self, model_path):
        with open(model_path, 'r') as model_file:
            model_specs = [model_spec.split() for model_spec in model_file.read().split('\n') if model_spec != '']
        
        self.model_components = []
        
        for model_spec in model_specs:
            if model_spec[0] == 'conv':
                self.model_components.append(ConvolutionLayer(num_filters=int(model_spec[1]), kernel_size=int(model_spec[2]), stride=int(model_spec[3]), padding=int(model_spec[4])))
            elif model_spec[0] == 'relu':
                self.model_components.append(ActivationLayer())
            elif model_spec[0] == 'max_pool':
                self.model_components.append(MaxPoolingLayer(kernel_size=int(model_spec[1]), stride=int(model_spec[2])))
            elif model_spec[0] == 'flatten':
                self.model_components.append(FlatteningLayer())
            elif model_spec[0] == 'fc':
                self.model_components.append(FullyConnectedLayer(output_dim=int(model_spec[1])))
            elif model_spec[0] == 'softmax':
                self.model_components.append(SoftmaxLayer())
            else:
                print('Unknown layer')
    
    def __str__(self):
        return '\n'.join(map(str, self.model_components))
    
    def train(self, u, y_true, lr):
        for i in range(len(self.model_components)):
            u = self.model_components[i].forward(u)
        
        del_v = u - y_true  # denoting y_predicted by u
        
        for i in range(len(self.model_components) - 1, -1, -1):
            del_v = self.model_components[i].backward(del_v, lr)
    
    def predict(self, u):
        for i in range(len(self.model_components)):
            u = self.model_components[i].forward(u)
        
        return u  # denoting y_predicted by u
    
    def save_model(self):
        for i in range(len(self.model_components)):
            self.model_components[i].save_learnable_parameters()
    
    def set_model(self):
        for i in range(len(self.model_components)):
            self.model_components[i].set_learnable_parameters()


# Performance Scorers Definition
def calculate_cross_entropy_loss(y_true, y_predicted):
    return np.sum(-1 * np.sum(y_true * np.log(y_predicted), axis=0))

def calculate_f1_scores(num_classes, y_true, y_predicted):
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    
    for i in range(y_true.shape[0]):
        if y_true[i, 0] == y_predicted[i, 0]:
            true_positives[y_true[i, 0]] = true_positives[y_true[i, 0]] + 1
        else:
            false_positives[y_predicted[i, 0]] = false_positives[y_predicted[i, 0]] + 1
            false_negatives[y_true[i, 0]] = false_negatives[y_true[i, 0]] + 1
    
    # ref: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
    accuracy = np.sum(true_positives) / (np.sum(true_positives) + 0.5 * (np.sum(false_positives) + np.sum(false_negatives)))  # micro/global average f1 score
    f1_score = np.mean(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))  # macro average f1 score
    return accuracy, f1_score

# Model Architecture Initialization
# ref: https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(0)
model = Model(model_path='data/NumtaDB_with_aug/models/lenet5.txt')

# Hyperparameters Configuration
# use_mnist = True  # use_mnist -> True: Use MNIST; False: Use CIFAR-10;
num_classes = 10

num_samples = 32
num_epochs = 10
lr = 0.001

# Model Training, Validation and Testing
# x_train, y_train, x_validation, y_validation, x_test, y_test = load_mnist_dataset() if use_mnist else load_cifar_10_dataset()
# x_train, y_train = subsample_dataset(num_classes, num_samples_per_class, x_train, y_train)
# x_validation, y_validation = subsample_dataset(num_classes, num_samples_per_class // 10, x_validation, y_validation)
# x_test, y_test = subsample_dataset(num_classes, num_samples_per_class // 10, x_test, y_test)

x_train, y_train = load_datasets(conf.train_image_folder, conf.train_labels)
y_true = np.zeros((10, y_train.shape[0]))
    
for i in range(y_true.shape[1]):
    y_true[y_train[i], i] = 1  # generating one-hot encoding of y_test

    

for epoch in range(conf.EPOCHS):
    print(epoch)
    model.train(x_train, y_true, lr)

y_pred = model.predict(x_train)

y_classes = np.argmax(y_pred, axis=0)

print(y_classes)
print(y_pred)

print('Correct:', np.sum(y_classes == y_train))





# num_batches = math.ceil(y_train.shape[0] / num_samples)
# min_f1_score = math.inf
# validation_stats = []

# for epoch in range(num_epochs):
#     for batch in range(num_batches):
#         print(f'(Training) Epoch: {epoch + 1} -> {batch + 1}/{num_batches} Batches Trained.', end='\r')
#         n_samples = y_train.shape[0] - batch * num_samples if (batch + 1) * num_samples > y_train.shape[0] else num_samples
#         y_true = np.zeros((num_classes, n_samples))
        
#         for i in range(y_true.shape[1]):
#             y_true[y_train[batch * num_samples + i, 0], i] = 1  # generating one-hot encoding of y_train
        
#         model.train(x_train[batch * num_samples: batch * num_samples + n_samples], y_true, lr)
#     print()
    
#     y_true = np.zeros((num_classes, y_validation.shape[0]))
#     y_predicted = model.predict(x_validation)
    
#     for i in range(y_true.shape[1]):
#         y_true[y_validation[i, 0], i] = 1  # generating one-hot encoding of y_validation
    
#     cross_entropy_loss = calculate_cross_entropy_loss(y_true, y_predicted)
#     accuracy, f1_score = calculate_f1_scores(num_classes, y_validation, np.reshape(np.argmax(y_predicted, axis=0), y_validation.shape))
    
#     if f1_score < min_f1_score:
#         min_f1_score = f1_score
#         model.save_model()
    
#     validation_stats.append([epoch + 1, cross_entropy_loss, accuracy, f1_score])
#     print(f'\n(Validation) Epoch: {epoch + 1} -> CE Loss: {cross_entropy_loss:.4f}\tAccuracy: {accuracy:.4f}\tF1 Score: {f1_score:.4f}\n')

# if not os.path.exists('outputdir/'):
#     os.makedirs('outputdir/')

# with open('outputdir/validation_stats.csv', 'w') as csv_file:
#     csv_writer = csv.writer(csv_file) 
#     csv_writer.writerow(['Epoch', 'CE Loss', 'Accuracy', 'F1 Score']) 
#     csv_writer.writerows(validation_stats)

# model.set_model()

# y_true = np.zeros((num_classes, y_test.shape[0]))
# y_predicted = model.predict(x_test)

# for i in range(y_true.shape[1]):
#     y_true[y_test[i, 0], i] = 1  # generating one-hot encoding of y_test

# cross_entropy_loss = calculate_cross_entropy_loss(y_true, y_predicted)
# accuracy, f1_score = calculate_f1_scores(num_classes, y_test, np.reshape(np.argmax(y_predicted, axis=0), y_test.shape))

# test_stats = [[cross_entropy_loss, accuracy, f1_score]]
# print(f'(Testing) -> CE Loss: {cross_entropy_loss:.4f}\tAccuracy: {accuracy:.4f}\tF1 Score: {f1_score:.4f}')

# with open('outputdir/test_stats.csv', 'w') as csv_file:
#     csv_writer = csv.writer(csv_file) 
#     csv_writer.writerow(['CE Loss', 'Accuracy', 'F1 Score']) 
#     csv_writer.writerows(test_stats)
