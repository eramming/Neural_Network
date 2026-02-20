from typing import List, Tuple
import numpy as np
from FC_NN import FC_NN

class Trainer:

    def __init__(self, nn: FC_NN, epochs: int, eta: float) -> None:
        self.epochs: int = epochs
        self.eta: float = eta
        self.nn: FC_NN = nn
    
    def train(self, inputs: List[float], target: np.ndarray) -> None:
        for epoch in self.epochs:
            self.ffbp(inputs)
            e = np.array([target[i] - p.output for i, p in enumerate(self.nn.p_layers[-1].perceptrons)])
            E = 0.5 * np.sum(np.square(e))
            print(f"Epoch {epoch}\tE:{E}")


    # Breadth First Traversal to compute Feed Forward Back Propagation algorithm!
    def ffbp(self, inputs: List[float]) -> None:
        for i, n in zip(inputs, self.nn.inputs):
            n.set_output(i)
        self.ffbp_helper(0)
        self.update_all_nodes()

    def ffbp_helper(self, layer_indx: int) -> None:
        if layer_indx >= len(self.nn.p_layers):
            return

        for p in self.nn.p_layers[layer_indx].perceptrons:
            p.set_output()
        self.ffbp_helper(layer_indx + 1)
        for p in self.nn.p_layers[layer_indx].perceptrons:
            p.store_update()


    def update_all_nodes(self) -> None:
        for layer in self.nn.p_layers:
            for p in layer.perceptrons:
                p.apply_update()

    # def back_propagation(self, outputs: np.ndarray) -> np.ndarray:
    #     error: float = self.target - outputs[-1]
    #     for level in range(len(self.nn.p_layers) - 1, -1, -1):
    #         self.update(level, error)
        
    #     self.nn.update()
    #     raise NotImplementedError()
    
    # def update(self, level: int, error: float) -> Tuple[np.ndarray, np.ndarray]:
    #     final_layer_indx: int = len(self.nn.p_layers) - 1
    #     if level == final_layer_indx:
    #         for p in self.nn.p_layers[level]:
    #             p.store_update(self.eta, error)
    #     elif level == final_layer_indx - 1:
    #         raise NotImplementedError("The current implementation can only handle NNs with up to one layer.")
    #     elif level < final_layer_indx - 1:
    #         raise NotImplementedError("The current implementation can only handle NNs with up to two layers.")

    # def ffbp(self, inputs: np.ndarray) -> np.ndarray:
    #     outputs = self.nn.feed_forward(inputs)
    #     self.back_propagation(outputs)
    #     return outputs[-1]