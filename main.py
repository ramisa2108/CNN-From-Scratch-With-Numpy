import numpy as np
from config import Config
from utils import *
from model import CNN

conf = Config()

if __name__ == '__main__':

    np.random.seed(0)

    # train_X, train_y = load_datasets(conf.train_image_folder, conf.train_labels)
    # test_X, test_y = load_datasets(conf.test_image_folder, conf.test_labels)
    
    # train_X, train_y = load_toy_dataset("data/Toy Dataset/trainNN.txt")
    # test_X, test_y = load_toy_dataset("data/Toy Dataset/testNN.txt")

    train_X, train_y_labels = load_datasets(conf.train_image_folder, conf.train_labels)
    train_y = one_hot_encoding(train_y_labels, 10)
    
    (train_X, train_y), (val_X, val_y) = split_train_set(train_X, train_y, conf.train_size, conf.val_size)

    model_desc = load_model_description(os.path.join(conf.model_folder, conf.model_desc_file))
    model = CNN(model_desc)

    
    for epoch in range(conf.epochs):
        print('Epoch:', epoch)
        y_out = model.forward(train_X)
        dy = model.backward(train_y, conf.learning_rate)

    y_out = model.forward(train_X)
    y_labels = np.argmax(y_out, axis=1)

    print(y_labels, y_labels.shape)
    print(train_y_labels)

    print(y_out)

    print('Correct:', np.sum(train_y_labels == y_labels))
    
    