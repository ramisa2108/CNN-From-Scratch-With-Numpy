import numpy as np
import pandas as pd
import os
import cv2
import pickle
from config import Config
conf = Config()

def one_hot_encoding(Y):

    m = Y.shape[0]
    one_hot = np.zeros((m, conf.num_labels))
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

    all_images = os.listdir(image_folder)

    if dataset_size is not None:
        np.random.shuffle(all_images)
        all_images = all_images[:dataset_size]

    X = []
    y = []

    for image_file in all_images:
        img = cv2.imread(os.path.join(image_folder, image_file), conf.read_colored_image)
        img = cv2.resize(img, (64, 64))
        img = (255.0 - img) / 255.0

        X.append(img)
        y.append(map[image_file])
        
    X = np.array(X)
    y = np.array(y)


    # for RGB
    if conf.read_colored_image:
        X = X.transpose(0, 3, 1, 2)
    # for Grayscale
    else:
        X = X[:, np.newaxis, :, :]

    
    print("{} images loaded from {}, labels loaded {}".format(len(X), image_folder, len(y)))
    return X, y


def load_model_description(file_name):

    with open(file_name, 'rb') as file:
        layers = pickle.load(file)

    return layers

def cross_entropy_loss(true_labels, output_propabilities):
    return np.sum(-1.0 * true_labels * np.log(output_propabilities))

def accuracy(true_labels, predicted_labels):
    return np.sum(true_labels == predicted_labels) / len(true_labels)


def macro_f1_score(true_labels, predicted_labels):

    f1_scores = []
    for l in range(conf.num_labels):
        pos = np.where(predicted_labels == l)[0]
        tp = np.sum(true_labels[pos] == l)
        fp = np.sum(true_labels[pos] != l)
        pos = np.where(true_labels == l)[0]
        fn = np.sum(predicted_labels[pos] != l)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0
        f1_scores.append(f1)
        
    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score

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
        

    

    


    

    
    

