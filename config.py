import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'

    train_image_folder = os.path.join(data_folder, 'train')
    test_image_folder = os.path.join(data_folder, 'test')
    train_labels = os.path.join(data_folder, 'train.csv')
    test_labels = os.path.join(data_folder, 'test.csv')

    train_size = 5000
    val_prop = 0.2
    test_size = 500


    model_folder = os.path.join(data_folder, 'models')
    model_desc_file = 'small_lenet5.pkl'
    model_weight_file = 'small_lenet5'
    load_pretrained = False

    num_labels = 10
    read_colored_image = 0
    learning_rate = 0.01
    batch_size = 32
    epochs = 20




        

