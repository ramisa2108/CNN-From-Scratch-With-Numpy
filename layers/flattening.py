import numpy as np

class Flattening:
    
    def forward(self, input):
        
        output = input.flatten()
        return output

    def backward(self, input, output):

        return 
    