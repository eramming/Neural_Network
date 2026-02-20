from typing import Callable, List
import numpy as np
# from functools import reduce
# import operator
from FC_NN import FC_NN
from Perceptron import Perceptron
from Perceptron import InputNode


class Layer:

    def __init__(self, perceptrons: List[Perceptron]) -> None:
        self.perceptrons: List[Perceptron] = perceptrons


class FC_NN:

    def __init__(self, layers: List[Layer], inputs: List[InputNode],
                 eta: float = 1.0) -> None:
        if len(layers) > 2:
            raise ValueError("The current implementation can only handle NNs with up to 2 layers.")
        self.inputs: List[InputNode] = inputs
        self.p_layers: List[Layer] = layers
        # self.weight_cnt: int = reduce(operator.add, [len(p.w_vec) for layer in self.p_layers for p in layer.perceptrons], 0)
        # self.perceptron_cnt: int = reduce(operator.add, [1 for layer in self.p_layers for p in layer.perceptrons], 0)

    def predict(self) -> List[float]:
        return [p.get_output() for p in self.p_layers[-1].perceptrons]

    @staticmethod
    def from_config(input_size: int, config: List[List[List[float]]],
                    f_act: Callable[[float], float]) -> 'FC_NN':
        layers: List[Layer] = FC_NN.create_perceptron_layers(config, f_act)
        inputs: List[InputNode] = [InputNode() for _ in range(input_size)]
        FC_NN.add_perceptron_inputs(layers, inputs)
        return FC_NN(layers, inputs)
    
    @staticmethod
    def create_perceptron_layers(config: List[List[List[float]]],
                                 f_act: Callable[[float], float]) -> List[Layer]:
        layers: List[Layer] = []
        for i, layer in enumerate(config):
            perceptrons: List[Perceptron] = []
            for j, p_weights in enumerate(layer):
                perceptrons.append(Perceptron(np.array(p_weights[:-1]),
                                              p_weights[-1], f_act, [], f"P_{i}{j}"))
            layers.append(Layer(perceptrons))
        return layers
    
    @staticmethod
    def add_perceptron_inputs(layers: List[Layer], inputs: List[InputNode]) -> None:
        layer_to_update: List[Perceptron] = layers[-1]
        for layer in reversed(layers[1:]):
            for p in layer_to_update:
                p.inputs = layer.perceptrons
            layer_to_update = layer
        for p in layers[0].perceptrons:
            p.inputs = inputs