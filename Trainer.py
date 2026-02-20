from typing import List
import numpy as np
from FC_NN import FC_NN


class Trainer:

    def __init__(self, nn: FC_NN, epochs: int, eta: float) -> None:
        self.epochs: int = epochs
        self.eta: float = eta
        self.nn: FC_NN = nn
    
    def train(self, inputs: List[float], targets: np.ndarray) -> None:
        for epoch in range(1, self.epochs + 1):
            self.ffbp(inputs, targets)
            e = np.array([targets[i] - p.output for i, p in enumerate(self.nn.p_layers[-1].perceptrons)])
            E = 0.5 * np.sum(np.square(e))
            print(f"Epoch {epoch}\tE:{E:.6f}\tResult:{[round(val, 6) for val in self.nn.predict()]}")


    # Breadth First Traversal to compute Feed Forward Back Propagation algorithm!
    def ffbp(self, inputs: List[float], targets: np.ndarray) -> None:
        for i, n in zip(inputs, self.nn.inputs):
            n.set_output(i)
        self.ffbp_helper(0, targets)
        self.update_all_nodes()

    def ffbp_helper(self, layer_indx: int, targets: np.ndarray) -> None:
        if layer_indx >= len(self.nn.p_layers):
            return

        for p in self.nn.p_layers[layer_indx].perceptrons:
            p.set_output()
        self.ffbp_helper(layer_indx + 1, targets)
        for i, p in enumerate(self.nn.p_layers[layer_indx].perceptrons):
            is_terminal: bool = False
            targ = None
            if layer_indx == len(self.nn.p_layers) - 1:
                is_terminal = True
                targ = targets[i]
            p.store_update(self.eta, targ, is_terminal)


    def update_all_nodes(self) -> None:
        for layer in self.nn.p_layers:
            for p in layer.perceptrons:
                p.apply_update()
