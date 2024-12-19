import random
from node import Node


class Optimizer:
    def __init__(self, params: list[Node], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0


class SGD(Optimizer):
    def __init__(self, params: list[Node], lr: float):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            p.value -= self.lr * p.grad


class AdamW(Optimizer):
    def __init__(
        self,
        params: list[Node],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    ):
        super().__init__(params, lr)

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [0.0 for _ in params]
        self.v = [0.0 for _ in params]

        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            p = self.params[i]
            b1, b2 = self.betas

            # Apply weight decay
            p.value -= self.lr * self.weight_decay * p.value

            # Update biased first moment estimate
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            # Update biased second moment estimate
            self.v[i] = b2 * self.v[i] + (1 - b2) * (p.grad**2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)

            # Update parameter value
            p.value -= self.lr * m_hat / (v_hat**0.5 + self.eps)

        # Increment timestep
        self.t += 1


class TanH:
    def __call__(self, x: Node):
        return x.tanh()


class Sigmoid:
    def __call__(self, x: Node):
        return x.sigmoid()


class ReLu:
    def __call__(self, x: Node):
        return x.relu()


class Neuron:
    def __init__(self, input_size: int, bias: bool = True, activation=None):
        limit = (1 / input_size) ** 0.5

        self.weights = [Node(random.uniform(-limit, limit)) for _ in range(input_size)]
        self.with_bias = bias
        self.bias = Node(0)
        self.activation = activation

    def __call__(self, x: list[float]):
        x = sum([w * x for w, x in zip(self.weights, x)]) + self.bias
        return self.activation(x) if self.activation else x

    def parameters(self):
        return self.weights + ([self.bias] if self.with_bias else [])


class Layer:
    def __init__(self, input_size: int, output_size: int, activation):
        print(f"Layer: {input_size} -> {output_size}, activation: {activation}")

        self.neurons = [
            Neuron(input_size, activation=activation) for _ in range(output_size)
        ]

    def __call__(self, x: list[float]):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, sizes: list[int], activation):
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=(activation if i < len(sizes) - 2 else None),
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        # minus max value for numerical stability
        max_val = max([xi.value for xi in x])
        exp = [(xi - max_val).exp() for xi in x]
        sum_exp = sum(exp)
        probabilities = [xi / sum_exp for xi in exp]

        return probabilities

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def save(self, path: str):
        try:
            with open(path, "w") as file:
                file.write(",".join([str(p.value) for p in self.parameters()]))
        except:
            print(f"Could not save model weights to file '{path}'")

    def load_state_dict(self, path: str):
        try:
            with open(path, "r") as file:
                loaded_weights = [float(p) for p in file.read().split(",")]

                if len(loaded_weights) != len(self.parameters()):
                    print(f"Model sizes don't match in file '{path}'")
                    return

                for p, w in zip(self.parameters(), loaded_weights):
                    p.value = float(w)

                print(f"Loaded weights from '{path}'")
        except:
            print(f"Could not load weights from file '{path}'")
