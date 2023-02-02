import numpy as np
from config import Config
import cv2
import os

conf = Config()

if __name__ == '__main__':

    img = cv2.imread(os.path.join(conf.train_folder, 'a00199.png'))
    