from typing import List
from Perceptron import Perceptron, InputNode
import numpy as np


class Layer:

    def __init__(self, perceptrons: List[Perceptron]) -> None:
        self.perceptrons: List[Perceptron] = perceptrons


class FC_NN:

    def __init__(self, layers: List[Layer], inputs: List[InputNode]) -> None:
        self.inputs: List[InputNode] = inputs
        self.p_layers: List[Layer] = layers

    # Breadth First Traversal to compute just Feed Forward outputs. No updating weights!
    def feed_forward(self, inputs: List[float]) -> List[float]:
        for i, n in zip(inputs, self.inputs):
            n.set_output(i)
        return self.ff_helper(0)

    def ff_helper(self, layer_indx: int) -> List[float]:
        for p in self.p_layers[layer_indx].perceptrons:
            p.set_output()
        if layer_indx == len(self.p_layers) - 1:
            return [p.get_output() for p in self.p_layers[layer_indx].perceptrons]
        return self.ff_helper(layer_indx + 1)
    
    def get_weights(self) -> List[List[List[float]]]:
        weights_by_layer = []
        for layer in self.p_layers:
            weights_by_perceptron = []
            for p in layer.perceptrons:
                weights_by_perceptron.append(np.append(p.w_vec, p.bias).tolist())
                
            weights_by_layer.append(weights_by_perceptron)
        return weights_by_layer