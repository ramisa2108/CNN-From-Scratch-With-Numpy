from abc import ABC, abstractclassmethod

class NNLayer(ABC):


    @abstractclassmethod
    def forward(self, X):
        pass

    @abstractclassmethod
    def backward(self, dZ, lr):
        pass

    def __str__(self):
        return "NNLayer"
