# Customizable Supervised Neural Netork

This repo was created for my JHU 605.647 Neural Networks course. It supports the creation of any depth of a fully connected network architecture, such that each layer is fully connected to both the previous and next layers but has no self connections. The network is trained using feed forward back propagation with gradient descent.


## Usage
You can specify the architecture of your network using "a-b-c" syntax. Each letter represents the number of perceptrons in that layer. Note, the first layer is the input layer and not perceptrons. For instance, "4-4-2" has four inputs to the first layer. That first layer has four perceptrons and the final layer has two perceptrons.

You can create the initial weights randomly using the TestCase class, or manually using the build_from_config() method of the FC_NNBuilder class.


## Training Methods
There are two methods to choose from for training. The first is the "online" approach; here, input-output pairs are switched with each iteration of the ffbp algorithm. This tends to lead to a more balanced final set of weights. The method is named "train_alternate" in the code.

The second method represents a more batched approach. It is named "train_delayed_alternate" since we switch back and forth between input-output pairs but only after training on a single pair for a bit. That amount is a parameter that can be configured. Technically, if you set it to 1, it would be the exact same as the method described above. In general, it is different and tends to skew the weights more towards the final input-output pair.