from typing import Callable, List
import numpy as np
from FC_NN import Layer, FC_NN
from Perceptron import InputNode, Perceptron
import random
from Activation import Activation
from logging import getLogger, Logger

LOG: Logger = getLogger(f"nn.{__name__}")


class FC_NNBuilder:

    def __init__(self, arch_str: str):
        self.arch: List[int] = list(map(int, arch_str.split("-")))
    
    def build_random(self) -> FC_NN:
        return self._create_nn()

    def _create_nn(self) -> FC_NN:
        config = []
        for i in range(1, len(self.arch)):
            layer = [[random.random() for _ in range(self.arch[i-1] + 1)] for _ in range(self.arch[i])]
            config.append(layer)
        LOG.debug(f"Config: {config}")
        return self.build_from_config(self.arch[0], config, Activation.sigmoid)

    def build_from_config(self, input_size: int, config: List[List[List[float]]],
                    f_act: Callable[[float], float]) -> FC_NN:
        layers: List[Layer] = self.create_perceptron_layers(config, f_act)
        inputs: List[InputNode] = [InputNode() for _ in range(input_size)]
        self.add_bidirectional_connections(layers, inputs)
        return FC_NN(layers, inputs)
    
    def create_perceptron_layers(self, config: List[List[List[float]]],
                                 f_act: Callable[[float], float]) -> List[Layer]:
        layers: List[Layer] = []
        for i, layer in enumerate(config):
            perceptrons: List[Perceptron] = []
            for j, p_weights in enumerate(layer):
                perceptrons.append(Perceptron(np.array(p_weights[:-1]),
                                              p_weights[-1], f_act, (i, j)))
            layers.append(Layer(perceptrons))
        return layers
    
    def add_bidirectional_connections(self, layers: List[Layer], inputs: List[InputNode]) -> None:
        consuming_layer: List[Perceptron] = layers[-1].perceptrons
        for producing_layer in [layer.perceptrons for layer in reversed(layers[:-1])]:
            for p in consuming_layer:
                p.set_producers(producing_layer)
            for p in producing_layer:
                p.set_consumers(consuming_layer)
            consuming_layer = producing_layer
        for p in layers[0].perceptrons:
            p.set_producers(inputs)