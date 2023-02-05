import numpy as np
from config import Config
from utils import *
from model import CNN

conf = Config()

if __name__ == '__main__':

    np.random.seed(0)

    train_X, train_y_labels = load_datasets(conf.train_image_folder, conf.train_labels)
    train_y = one_hot_encoding(train_y_labels, 10)
    
    test_X, test_y_labels = load_datasets(conf.test_image_folder, conf.test_labels)
    test_y = one_hot_encoding(test_y_labels, 10)
    
    (train_X, train_y), (val_X, val_y) = split_train_set(train_X, train_y, conf.train_size, conf.val_size)

    model_desc = load_model_description(os.path.join(conf.model_folder, conf.model_desc_file))
    model = CNN(model_desc)

    
    for epoch in range(conf.epochs):

        train_loss = 0.0
        for i in range(0, train_X.shape[0], conf.batch_size):
            
            train_X_batch = train_X[i: i+conf.batch_size]
            train_y_batch = train_y[i: i+conf.batch_size]   
            print(train_X_batch.shape) 
            y_out = model.forward(train_X_batch)
            dy = model.backward(train_y_batch, conf.learning_rate)
            train_loss += np.sum(-1.0 * train_y_batch * np.log(y_out))
        
        train_loss /= train_X.shape[0]
        print('Epoch:', epoch, 'train loss:', train_loss)
        
    y_predicted, y_predicted_labels=model.predict(test_X)
    
    print('test loss:', np.sum(-1.0 * test_y * np.log(y_predicted))/len(test_y))
    print('test accuracy', np.sum(test_y_labels == y_predicted_labels)/len(test_y_labels) * 100, "%")
    
    