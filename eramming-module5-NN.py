from FC_NN import FC_NN
from Activation import Activation
from Trainer import Trainer
from typing import List


def main() -> None:
    init_weights = [
        [[0.7, 0.7, 0.7, 0], [0.4, 0.4, 0.4, 0]],
        [[0.5, 0.1, 0], [0.5, 0.1, 0], [0.5, 0.1, 0]],
        [[0.3, 0.5, 0.7, 0], [0.4, 0.6, 0.8, 0]]
    ]
    nn: FC_NN = FC_NN.from_config(3, init_weights, Activation.sigmoid)
    trainer = Trainer(5, 1.0, nn)
    inputs = [1.9, 7, -2.2]
    targets = [0.14]
    trainer.train(inputs, targets)
    prediction: List[float] = nn.predict()
    print(f"Desired Value: {targets}\tPredicted Value(s): {prediction}")


if __name__ == "__main__":
    main()
