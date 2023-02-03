import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'

    train_image_folder = os.path.join(data_folder, 'train_small')
    test_image_folder = os.path.join(data_folder, 'test_small')
    train_labels = os.path.join(data_folder, 'train_small.csv')
    test_labels = os.path.join(data_folder, 'test_small.csv')

    train_size = 500
    val_size = 5
    test_size = 5

    learning_rate = 0.001
    batch_size = 16
    epochs = 11

    model_folder = os.path.join(data_folder, 'models')
    model_desc_file = 'linear_model_desc.pkl'
    model_weight_file = None



        

