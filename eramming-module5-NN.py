from FC_NN import FC_NN
from Activation import Activation
from Trainer import Trainer
from typing import List
import numpy as np
import random
from FC_NNBuilder import FC_NNBuilder
from argparse import ArgumentParser, Namespace
from logging import getLogger, Logger, DEBUG, INFO, basicConfig
import sys

random.seed(777)
basicConfig(
    level=INFO
)
LOG: Logger = getLogger(f"nn.{__name__}")

class TestCase:

    def __init__(self, arch_str: str) -> None:
        self.arch: List[int] = list(map(int, arch_str.split("-")))
        self.input_set: List[float] = self._create_inputs(self.arch[0])
        self.target_set: np.ndarray = self._create_targets(self.arch[-1])
        self.nn: FC_NN = FC_NNBuilder(arch_str).build_random()

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

def parse_args(argv: List[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(description="Create a Fully Connected Neural Network.")
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs to train.")
    parser.add_argument("--eta", type=float, required=False, help="Training rate.")
    parser.add_argument("--debug", action="store_true", required=False, help="Show debug statements")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    # Random Test Case for Sanity Testing:
    # tc = TestCase("2-2-1")
    # nn: FC_NN = tc.nn
    # inputs = [tc.input_set]
    # targets = [tc.target_set]

    # Overrides for Class Assignment:
    inputs = [[1, 1], [-1, -1]]
    targets = [np.array([0.9]), np.array([0.05])]
    config = [
        [[0.3, 0.3, 0], [0.3, 0.3, 0]],
        [[0.8, 0.8, 0]]
    ]
    nn: FC_NN = FC_NNBuilder("2-2-1").build_from_config(len(inputs[0]), config, Activation.sigmoid)
    args: Namespace = parse_args(argv[1:])
    epochs: int = 15
    eta: float = 1.0
    if args.epochs is not None:
        epochs = args.epochs
    if args.eta is not None:
        eta = args.eta
    if args.debug:
        getLogger("nn").setLevel(DEBUG)

    LOG.info("------------Single Alternating---------------")
    trainer = Trainer(nn, 15, eta, inputs, targets)
    LOG.info(f"Inputs: {inputs}\tTargets: {targets}")
    trainer.train_alternate()
    for input, target in zip(inputs, targets):
        outputs: List[float] = nn.feed_forward(input)
        e = np.array([target[0] - outputs[0]])
        E = 0.5 * np.sum(np.square(e))
        weights_ext = nn.get_weights()
        LOG.info(f"\tDesired Value: {target}\tPredicted Value(s): {outputs}\tE:{E}")
    LOG.info(f"\tFinal Weights: {weights_ext}")

    LOG.info("------------Delayed Alternating---------------")
    nn: FC_NN = FC_NNBuilder("2-2-1").build_from_config(len(inputs[0]), config, Activation.sigmoid)
    trainer = Trainer(nn, epochs, eta, inputs, targets)
    trainer.train_delayed_alternate(switch_after=15)
    for input, target in zip(inputs, targets):
        outputs: List[float] = nn.feed_forward(input)
        e = np.array([target[0] - outputs[0]])
        E = 0.5 * np.sum(np.square(e))
        weights_ext = nn.get_weights()
        LOG.info(f"\tDesired Value: {target}\tPredicted Value(s): {outputs}\tE:{E}")
    LOG.info(f"\tFinal Weights: {weights_ext}")


if __name__ == "__main__":
    main(sys.argv)
