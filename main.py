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


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title(title)
    plt.savefig(os.path.join(conf.output_folder, title))
    plt.clf()


def train_and_test():

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
    plot_confusion_matrix(val_y_labels, val_pred_labels, 'Confusion Matrix For Validation Set')
    
    print('Training completed. Testing...')

    test_X, test_y_labels = load_datasets(conf.test_image_folder, conf.test_labels, conf.test_size)
    test_y = one_hot_encoding(test_y_labels)

    test_pred = eval_epoch(model, test_X)
    test_loss, test_f1_score, test_accuracy = calculate_metrics(test_y, test_pred)

    print('Test acc: {} loss: {} , F1: {}'.format(test_accuracy, test_loss, test_f1_score))

    test_y_labels = get_labels(test_y)
    test_pred_labels = get_labels(test_pred)
    plot_confusion_matrix(test_y_labels, test_pred_labels, 'Confusion Matrix For Test Set')
    
    with open(os.path.join(conf.output_folder, 'test_file_list.pickle'), 'rb') as file:
        test_file_names = pickle.load(file)

    output_df = pd.DataFrame(columns=['FileName', 'Digit'])
    output_df['FileName'] = test_file_names
    output_df['Digit'] = test_pred_labels
    output_df.to_csv(os.path.join(conf.output_folder, 'predictions.csv'), index=False)

def predict():

    model_weights = load_model_weights(conf.model_weight_file)
    model = CNN(model_weights)

    while True:
            image_path = input('Enter image path: ')
            X = load_image(image_path)
            X = np.array([X])
            
            # for RGB
            if conf.READ_COLORED_IMAGE:
                X = X.transpose(0, 3, 1, 2)
            # for Grayscale
            else:
                X = X[:, np.newaxis, :, :]
            
            prediction = eval_epoch(model, X)
            predicted_label = get_labels(prediction)[0]
            print('predicted digit:', predicted_label)
    

if __name__ == '__main__':

    np.random.seed(0)
    np.set_printoptions(precision=2)

    if conf.train_and_test_mode:
        train_and_test()
    else:
         predict()   
         
            

        


    