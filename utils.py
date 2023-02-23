import numpy as np
import pandas as pd
import os
import cv2
import pickle
from config import Config
from sklearn import metrics


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

def load_image(image_path):
    img = cv2.imread(image_path, conf.READ_COLORED_IMAGE)
    img = cv2.resize(img, conf.IMAGE_DIMS)
    img = (255.0 - img) / 255.0
    return img


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
        img = load_image(os.path.join(image_folder, image_file))
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
    
    if 'train' in image_folder:
        list_name = 'train_file_list.pickle'
    elif 'val' in image_folder:
        list_name = 'val_file_list.pickle'
    else:
        list_name = 'test_file_list.pickle'
        
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

