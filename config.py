import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'
    output_folder = 'outputs'

    train_image_folder = os.path.join(data_folder, 'train')
    test_image_folder = os.path.join(data_folder, 'test')
    train_labels = os.path.join(data_folder, 'train.csv')
    test_labels = os.path.join(data_folder, 'test.csv')

    train_size = None
    val_prop = 0.2
    test_size = None


    model_folder = 'saved_models'
    model_desc_file = 'small_lenet5_desc.pkl'
    model_weight_file = 'small_lenet5_model_weights.pkl'
    load_pretrained = False

    NUM_LABELS = 10
    READ_COLORED_IMAGE = 0
    LEARNING_RATE = 0.01
    IMAGE_DIMS = (32, 32)
    BATCH_SIZE = 32
    EPOCHS = 50


