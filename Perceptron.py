from typing import Callable, List, override, Tuple
import numpy as np
from abc import ABC, abstractmethod
from logging import getLogger, Logger

LOG: Logger = getLogger(f"nn.{__name__}")


class Node(ABC):

    @abstractmethod
    def get_output(self) -> float:
        pass

    @abstractmethod
    def set_output(self) -> None:
        pass


class InputNode(Node):

    def __init__(self, constant: float = 0):
        self.constant = constant

    @override
    def get_output(self) -> float:
        return self.constant
    
    @override
    def set_output(self, constant: float) -> None:
        self.constant = constant


class Perceptron(Node):

    def __init__(self, init_weights: np.ndarray, init_bias: float,
                 f_act: Callable[[float], float], loc: Tuple[int, int]) -> None:
        super().__init__()
        self.name: str = f"P_{loc[0]}_{loc[1]}"
        self.loc: Tuple[int, int] = loc
        self.f_activation: Callable[[float], float] = f_act
        self.w_vec: np.ndarray = init_weights
        self.bias: float = init_bias
        self.producers: List[Node] = []
        self.consumers: List[Perceptron] = []

        self.output: float = None
        self.new_weights_extended: np.ndarray = None
        self.delta: float = None


    def calculate_activity(self) -> float:
        inputs: np.ndarray = np.array([node.get_output() for node in self.producers])
        return np.dot(np.append(self.w_vec, self.bias), np.append(inputs, 1))
    
    def calculate_output(self) -> float:
        return self.f_activation(self.calculate_activity())
    
    def store_update(self, eta: float, target: float, is_terminal: bool) -> None:
        if is_terminal:
            e: float = target - self.output
            self.delta = e * (1 - self.output) * self.output
        else:
            self.delta = self.sum_downstream_weighted_deltas() * (1 - self.output) * self.output

        b_delta: float = eta * np.array([self.delta])
        ins = [p.get_output() for p in self.producers]
        delta_extended = np.append(b_delta * np.array(ins), b_delta)
        self.new_weights_extended = delta_extended + np.append(self.w_vec, self.bias)


    def sum_downstream_weighted_deltas(self) -> float:
        sum: float = 0
        for c in self.consumers:
            sum += c.delta * c.w_vec[self.loc[1]]
        return sum
        
        
    def apply_update(self) -> None:
        self.w_vec = self.new_weights_extended[:-1]
        self.bias = float(self.new_weights_extended[-1])
        self.new_weights_extended = None

    @override
    def set_output(self) -> None:
        self.output = self.calculate_output()
        LOG.debug(f"\tWeights: {self.w_vec}\tBias: {self.bias:.6f}\tOutput: {self.output:.6f}")
    
    @override
    def get_output(self) -> float:
        return self.output

    def set_producers(self, producers: List[Node]) -> None:
        self.producers = producers

    def set_consumers(self, consumers: List['Perceptron']) -> None:
        self.consumers = consumers