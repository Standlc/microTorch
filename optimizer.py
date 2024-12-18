from node import Node


class Optimizer:
    def __init__(self, params: list[Node], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self):
        for p in self.params:
            p.value -= self.lr * p.grad
