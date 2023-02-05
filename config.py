import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'

    train_image_folder = os.path.join(data_folder, 'train_small')
    test_image_folder = os.path.join(data_folder, 'test_small')
    train_labels = os.path.join(data_folder, 'train_small.csv')
    test_labels = os.path.join(data_folder, 'test_small.csv')

    train_size = 500
    val_size = 5
    test_size = 50

    learning_rate = 0.001
    batch_size = 32
    epochs = 10

    model_folder = os.path.join(data_folder, 'models')
    model_desc_file = 'lenet5.pkl'
    model_weight_file = None

    read_colored_image = 0



        

