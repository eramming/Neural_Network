from typing import Callable, List
import numpy as np
from Perceptron import Perceptron
from Perceptron import InputNode


class Layer:

    def __init__(self, perceptrons: List[Perceptron]) -> None:
        self.perceptrons: List[Perceptron] = perceptrons


class FC_NN:

    def __init__(self, layers: List[Layer], inputs: List[InputNode],
                 eta: float = 1.0) -> None:
        self.inputs: List[InputNode] = inputs
        self.p_layers: List[Layer] = layers

    def predict(self) -> List[float]:
        return [p.get_output() for p in self.p_layers[-1].perceptrons]

    @staticmethod
    def from_config(input_size: int, config: List[List[List[float]]],
                    f_act: Callable[[float], float]) -> 'FC_NN':
        layers: List[Layer] = FC_NN.create_perceptron_layers(config, f_act)
        inputs: List[InputNode] = [InputNode() for _ in range(input_size)]
        FC_NN.add_bidirectional_connections(layers, inputs)
        return FC_NN(layers, inputs)
    
    @staticmethod
    def create_perceptron_layers(config: List[List[List[float]]],
                                 f_act: Callable[[float], float]) -> List[Layer]:
        layers: List[Layer] = []
        for i, layer in enumerate(config):
            perceptrons: List[Perceptron] = []
            for j, p_weights in enumerate(layer):
                perceptrons.append(Perceptron(np.array(p_weights),
                                              f_act, (i, j)))
            layers.append(Layer(perceptrons))
        return layers
    
    @staticmethod
    def add_bidirectional_connections(layers: List[Layer], inputs: List[InputNode]) -> None:
        consuming_layer: List[Perceptron] = layers[-1].perceptrons
        for producing_layer in [layer.perceptrons for layer in reversed(layers[:-1])]:
            for p in consuming_layer:
                p.set_producers(producing_layer)
            for p in producing_layer:
                p.set_consumers(consuming_layer)
            consuming_layer = producing_layer
        for p in layers[0].perceptrons:
            p.set_producers(inputs)