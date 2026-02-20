import math


class Activation:

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0/(1.0+math.exp(-x))
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0, x)
    
    @staticmethod
    def leaky_relu(x: float) -> float:
        c: float = 0.1
        if x < 0:
            return c * x
        return x