from FC_NN import FC_NN
from Activation import Activation
from Trainer import Trainer
from typing import List
import numpy as np
import random

random.seed(777)


class TestCase:

    def __init__(self, arch_str: str) -> None:
        self.arch: List[int] = list(map(int, arch_str.split("-")))
        self.inputs: List[float] = self._create_inputs(self.arch[0])
        self.targets: np.ndarray = self._create_targets(self.arch[-1])
        self.nn: FC_NN = self._create_nn()

    def _create_nn(self) -> FC_NN:
        config = []
        for i in range(1, len(self.arch)):
            layer = [[random.random() for _ in range(self.arch[i-1] + 1)] for _ in range(self.arch[i])]
            config.append(layer)
        print(f"Config: {config}")
        return FC_NN.from_config(self.arch[0], config, Activation.sigmoid)

    def _create_inputs(self, in_size: int) -> List[float]:
        sample = [1.9, 7, -2.2, -8.4, 3.9]
        if in_size > len(sample):
            raise ValueError(f"Current input max is {len(sample)}")
        return sample[:in_size]

    def _create_targets(self, terminal_cnt: int) -> np.ndarray:
        sample = np.array([0.14, 0.33, 0.79, 0.5, 0.01])
        if terminal_cnt > len(sample):
            raise ValueError(f"Current terminal count max is {len(sample)}")
        return sample[:terminal_cnt]



def main() -> None:
    tc = TestCase("2-1")
    epochs: int = 5
    eta: float = 5.0

    # Overrides for Class Assignment:
    tc.nn = FC_NN.from_config(2, [[[0.24, 0.88, 0]]], Activation.sigmoid)
    tc.inputs = [0.8, 0.9]
    tc.targets = np.array([0.95])

    trainer = Trainer(tc.nn, epochs, eta)


    print(f"Inputs: {tc.inputs}\tTargets: {tc.targets}")
    trainer.train(tc.inputs, tc.targets)
    prediction: List[float] = tc.nn.predict()
    print(f"Desired Value: {tc.targets}\tPredicted Value(s): {prediction}")


if __name__ == "__main__":
    main()
