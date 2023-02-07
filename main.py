import numpy as np
from config import Config
from utils import *
from model import CNN
from tqdm import tqdm
import time

conf = Config()

def eval_epoch(model, val_X, val_y, is_test=False):
    
    n_val_batches = (val_X.shape[0] + conf.BATCH_SIZE - 1) // conf.BATCH_SIZE

    val_loss = 0.0
    val_pred_labels = np.array([])

    for i in tqdm(range(n_val_batches)):
        val_X_batch = val_X[i * conf.BATCH_SIZE : (i+1) * conf.BATCH_SIZE]
        val_y_batch = val_y[i * conf.BATCH_SIZE : (i+1) * conf.BATCH_SIZE]
        y_out_prob = model.predict(val_X_batch)
        val_loss += cross_entropy_loss(val_y_batch, y_out_prob)
        val_pred_labels = np.append(val_pred_labels, get_labels(y_out_prob))

    
    val_loss /= val_X.shape[0]
    val_y_labels = get_labels(val_y)
    val_f1_score = macro_f1_score(val_y_labels, val_pred_labels)
    val_accuracy = accuracy(val_y_labels, val_pred_labels)

    if is_test:
        label_wise_score(val_y_labels, val_pred_labels)
    
    return val_loss, val_f1_score, val_accuracy


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
    

if __name__ == '__main__':

    np.random.seed(0)
    np.set_printoptions(precision=2)

    train_X, train_y_labels = load_datasets(conf.train_image_folder, conf.train_labels, conf.train_size)
    train_y = one_hot_encoding(train_y_labels)

    (train_X, train_y), (val_X, val_y) = split_train_set(train_X, train_y, conf.val_prop)

    
    test_X, test_y_labels = load_datasets(conf.test_image_folder, conf.test_labels, conf.test_size)
    test_y = one_hot_encoding(test_y_labels)
    
    
    model_desc = load_model_description(os.path.join(conf.model_folder, conf.model_desc_file))
    model_weights = None
    if conf.load_pretrained:
        model_weights = load_model_weights(os.path.join(conf.model_folder, conf.model_weight_file))

    model = CNN(model_desc, model_weights)


    train_losses = []
    val_losses = []
    best_val_f1_score = 0.0
    for epoch in range(conf.EPOCHS):
        t1 = time.time()
        train_loss = train_epoch(model, train_X, train_y)
        val_loss, val_f1_score, val_accuracy = eval_epoch(model, val_X, val_y)
        t2 = time.time()
        print('Epoch {}: val_acc: {}  val_F1: {} val_loss: {} train_loss: {} time: {}'.format(epoch, val_accuracy, val_f1_score, val_loss, train_loss,t2-t1))

        if val_f1_score > best_val_f1_score:
            best_val_f1_score = val_f1_score
            model.save_model_weights(conf.model_folder, conf.model_weight_file)

        train_losses += [train_loss]
        val_losses += [val_loss]

    print('\n\n')
    print('Train Losses:', train_losses)
    print('Val Losses', val_losses)

    print('Training completed. Testing...')
    test_loss, test_f1_score, test_accuracy = eval_epoch(model, test_X, test_y, is_test=True)
    
    
    print('Test acc: {} loss: {} , F1: {}'.format(test_accuracy, test_loss, test_f1_score))
    