import os

class Config(object):

    data_folder = 'data/NumtaDB_with_aug'

    train_image_folder = os.path.join(data_folder, 'train')
    test_image_folder = os.path.join(data_folder, 'test')
    train_labels = os.path.join(data_folder, 'train.csv')
    test_labels = os.path.join(data_folder, 'test.csv')

    train_size = 5000
    val_prop = 0.2
    test_size = 5000


    model_folder = 'saved_models'
    model_weight_file = os.path.join(model_folder, 'big_model_weights.pkl')
    load_pretrained = False

    output_folder = 'outputs'

    NUM_LABELS = 10
    READ_COLORED_IMAGE = 0
    LEARNING_RATE = 0.01
    IMAGE_DIMS = (32, 32)
    BATCH_SIZE = 32
    EPOCHS = 20
