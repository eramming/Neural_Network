import math
from typing import Callable, List
import numpy as np
from functools import reduce
import operator


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


class Perceptron:

    def __init__(self, init_weights: np.ndarray, init_bias: float,
                 f_act: Callable[[float], float], name: str = "Perceptron") -> None:
        self.name = name
        self.f_activation: Callable[[float], float] = f_act
        self.w_vec: np.ndarray = init_weights
        self.bias = init_bias


    def calculate_activity(self, inputs: np.ndarray) -> float:
        if np.shape(inputs) != np.shape(self.w_vec):
            raise ValueError(f"Inputs and weights not the same dimensionality for {self.name}!"
                             f"\nInput: {np.shape(inputs)}\tWeights:{np.shape(self.w_vec)}")
        return np.dot(np.append(self.w_vec, self.bias), np.append(inputs, 1))
    
    def calculate_output(self, inputs: np.ndarray) -> float:
        return self.f_activation(self.calculate_activity(inputs))



class Layer:

    def __init__(self, perceptrons: List[Perceptron]) -> None:
        self.perceptrons: List[Perceptron] = perceptrons



class FC_NN:

    def __init__(self, layers: List[Layer]) -> None:
        if len(layers) > 2:
            raise ValueError("The current implementation can only handle NNs with up to 2 layers.")
        self.layers: List[Layer] = layers
        self.weight_cnt: int = reduce(operator.add, [len(p.w_vec) for layer in self.layers for p in layer.perceptrons], 0)
        self.perceptron_cnt: int = reduce(operator.add, [1 for layer in self.layers for p in layer.perceptrons], 0)

    def predict(self, inputs: np.ndarray) -> float:
        return self.feed_forward(inputs)[-1]

    def feed_forward(self, inputs: np.ndarray) -> List[float]:
        w_delta: np.ndarray = np.zeros(self.weight_cnt)
        bias_delta: np.ndarray = np.zeros(self.perceptron_cnt)
        return self.ff_helper(inputs, 0)

    def ff_helper(self, inputs: np.ndarray, layer_indx: int) -> List[float]:
        if layer_indx >= len(self.layers):
            return []

        layer_outputs: List[float] = []
        for p in self.layers[layer_indx].perceptrons:
            layer_outputs.append(p.calculate_output(inputs))
        return layer_outputs + self.ff_helper(layer_outputs, layer_indx + 1)


    @staticmethod
    def from_config(config: List[List[List[float]]], f_act: Callable[[float], float]) -> 'FC_NN':
        layers: List[Layer] = []
        for i, layer in enumerate(config):
            perceptrons: List[Perceptron] = []
            for j, p_weights in enumerate(layer):
                perceptrons.append(Perceptron(np.array(p_weights[:-1]),
                                              p_weights[-1], f_act, f"P_{i}{j}"))
            layers.append(Layer(perceptrons))
        return FC_NN(layers)



class Trainer:

    def __init__(self, nn: FC_NN, epochs: int, eta: float, target: float) -> None:
        self.epochs: int = epochs
        self.eta: float = eta
        self.target: float = target
        self.nn: FC_NN = nn
    
    def train(self, inputs: List[float], target: float) -> None:
        for epoch in self.epochs:
            outputs: List[float] = self.nn.feed_forward(np.array(inputs))
            error: float = target - outputs[-1]
            # w_delta: np.ndarray = self.eta * A

    def back_propagation(self, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        outputs = reversed(outputs)
        out_indx: int = 0
        w_deltas: list = []
        b_deltas: list = []
        for layer_indx, layer in enumerate(reversed(self.nn.layers)):
            w_deltas = w_deltas + self.weight_updates(layer_indx)
            b_deltas = b_deltas + self.bias_updates(layer_indx)
            out_indx += len(layer)
            
        raise NotImplementedError()
    
    def weight_updates(self, level: int) -> List[float]:
        if level == 0:
            error: float = self.target - outputs[-1]
            delta: float = error * (1 - outputs[-1]) * outputs[-1]
            w_deltas: np.ndarray = eta * delta * np.array([])
        elif level == 1:
            pass
        elif level > 1:
            ValueError("The current implementation can only handle NNs with up to two layers.")
    
    def weight_updates(self, level: int) -> List[float]:
        if level == 0:
            error: float = self.target - outputs[-1]
            delta: float = error * (1 - outputs[-1]) * outputs[-1]
            w_deltas: np.ndarray = eta * delta * np.array([])
        elif level == 1:
            pass
        elif level > 1:
            ValueError("The current implementation can only handle NNs with up to two layers.")

    def ffbp(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def main() -> None:
    nn: FC_NN = FC_NN.from_config([
        [[0.7, 0.7, 0.7, 0], [0.4, 0.4, 0.4, 0]],
        [[0.5, 0.1, 0], [0.5, 0.1, 0], [0.5, 0.1, 0]],
        [[0.3, 0.5, 0.7, 0], [0.4, 0.6, 0.8, 0]]
    ])
    trainer = Trainer(5, 1.0, nn)
    inputs = [1.9, 7, -2.2]
    target = 0.14
    trainer.train(inputs, target)
    prediction: float = nn.predict(inputs)
    print(f"Desired Value: {target}\tPredicted Value: {prediction}")


if __name__ == "__main__":
    main()
