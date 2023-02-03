import numpy as np
import pandas as pd
import os
import cv2
import pickle


def one_hot_encoding(Y, labels):

    m = Y.shape[0]
    one_hot = np.zeros((m, labels))
    one_hot[np.arange(m), Y] = 1
    return one_hot


def split_train_set(X, y, train_size, val_size):
    
    train_X, val_X = X[:train_size], X[train_size: train_size+val_size]
    train_y, val_y = y[:train_size], y[train_size: train_size+val_size]

    return (train_X, train_y), (val_X, val_y)

def load_datasets(image_folder, label_file):

    labels = pd.read_csv(label_file)
    map = dict(zip(labels['filename'], labels['digit']))

    all_images = os.listdir(image_folder)

    X = []
    y = []

    for image_file in all_images:
        img = cv2.imread(os.path.join(image_folder, image_file))

        img = img / 255.0

        X.append(img)
        y.append(map[image_file])
    
    X = np.array(X)
    y = np.array(y)

    
    print("{} images loaded from {}, labels loaded {}".format(len(X), image_folder, len(y)))
    return X, y


def load_model_description(file_name):

    with open(file_name, 'rb') as file:
        layers = pickle.load(file)

    return layers


def load_toy_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    X = []
    y = []
    for l in lines:
        xy = l.strip().split()
        y += [int(xy[-1])]
        X += [[float(x) for x in xy[:-1]]]

    X = np.array(X)
    X = X.reshape((-1, 2, 2))
    y = np.array(y)
    y = one_hot_encoding(y, 10)
    return X, y
        

    


    


    

    
    

