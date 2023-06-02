# CNN From Scratch With Numpy

Implementation of Convolutional Neural Network for image classification. The CNN model is written from scratch without any deep learning frameworks, using only Numpy library functions. The code has been vectorized for speeding up training and inference. 

### Dataset
Numta Handwritten Bangla Digits dataset. Dataset description can be found in [Kaggle](https://www.kaggle.com/competitions/numta/data)

Download the data from [here](https://bengali.ai/wp-content/uploads/datasets/NumtaDB_with_aug.zip) or [here](https://drive.google.com/drive/folders/1iaLxuSN88OyOuHwEbwBzfmwf9gFi_Vqn?usp=share_link).

Training and validation of the CNN model was performed using the images from training-a, training-b and training-c folders of the dataset. Training-d was used as the independent test set.

The dataset directory structure should be as follows:
```
data
├── NumtaDB_with_aug
│   ├── train
│   │   ├── *.png
│   ├── test
│   │   ├── *.png
│   ├── train.csv
│   ├── test.csv
```
The csv files contain the labels for the images in the train and test folders.

### Running the model
For training and testing the model, set *train_and_test_model* to True in config.py. The model will be trained and tested on the dataset. The trained model will be saved in the *saved_models* directory. For classifying individual images, set *train_and_test_model* to False and *load_pretrained* to True. The model will be loaded from the *saved_models* directory and the image will be classified.


### Useful articles

1. Back propagation for covolution layer with stride > 1 [part1](https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710) and [part2] (https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa)

2. Functions for vectorization:

 - numpy.einsum function [doc](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) and [guide](https://ajcr.net/Basic-guide-to-einsum/)
 -  numpy.lib.stride_tricks.as_strided function [doc](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)

3. Creating a Vectorized Convolution layer in NumPy [blogpost](https://blog.ca.meron.dev/Vectorized-CNN/)

