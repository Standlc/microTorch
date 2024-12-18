import random
from node import Node


class Neuron:
    def __init__(self, input_size: int, with_bias: bool = True, linear=False):
        limit = (6 / input_size) ** 0.5
        # limit = 1

        self.weights = [Node(random.uniform(-limit, limit)) for _ in range(input_size)]
        self.with_bias = with_bias
        self.bias = Node(0)
        self.is_linear = linear

    def __call__(self, x: list[float]):
        res = sum([w * x for w, x in zip(self.weights, x)]) + self.bias
        return res if self.is_linear else res.tanh()

    def parameters(self):
        if self.with_bias:
            return self.weights + [self.bias]
        return self.weights


class Layer:
    def __init__(self, input_size: int, output_size: int, linear=False):
        self.neurons = [
            Neuron(input_size, with_bias=True, linear=linear)
            for _ in range(output_size)
        ]

    def __call__(self, x: list[float]):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, sizes: list[int]):
        self.layers = [
            Layer(sizes[i], sizes[i + 1], linear=(i == len(sizes) - 2))
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        max_val = max([xi.value for xi in x])
        exp = [(xi - max_val).exp() for xi in x]
        sum_exp = sum(exp)
        probabilities = [xi / sum_exp for xi in exp]
        # print(probabilities)
        return probabilities

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
