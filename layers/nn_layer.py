from abc import ABC, abstractclassmethod

class NNLayer(ABC):

    
    def __call__(self, X):
        self.forward(X)

    @abstractclassmethod
    def forward(self, X):
        pass

    @abstractclassmethod
    def backward(self, dY):
        pass
