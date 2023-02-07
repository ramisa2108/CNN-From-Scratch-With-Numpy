from abc import ABC, abstractclassmethod

class NNLayer(ABC):


    @abstractclassmethod
    def forward(self, X):
        pass

    @abstractclassmethod
    def backward(self, dZ, lr):
        pass

    @abstractclassmethod
    def load_params(self):
        pass

    @abstractclassmethod
    def save_params(self):
        pass

    
    def __str__(self):
        return "NNLayer"
