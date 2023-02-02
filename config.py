import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'
    train_folder = os.path.join(data_folder, 'train_small')
    test_folder = os.path.join(data_folder, 'test_small')

        

