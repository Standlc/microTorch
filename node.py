import math


class Node:
    def __init__(self, value: float, prev: set["Node"] = ()):
        self.value = float(value)
        self.grad = 0
        self._prev = set(prev)
        self._backward = lambda: None

    def item(self):
        return self.value

    def __add__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward

        return out

    def __pow__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value**other.value, (self, other))

        def backward():
            self.grad += other.value * self.value ** (other.value - 1) * out.grad

        out._backward = backward

        return out

    def __mul__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other))

        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = backward

        return out

    def tanh(self) -> "Node":
        out = Node(math.tanh(self.value), (self,))

        def backward():
            self.grad += (1 - out.value**2) * out.grad

        out._backward = backward

        return out

    def sigmoid(self) -> "Node":
        out = Node(1 / (1 + math.exp(-self.value)), (self,))

        def backward():
            self.grad += out.value * (1 - out.value) * out.grad

        out._backward = backward

        return out

    def relu(self) -> "Node":
        out = Node(max(0, self.value), (self,))

        def backward():
            self.grad += (self.value > 0) * out.grad

        out._backward = backward

        return out

    def exp(self) -> "Node":
        try:
            exp = math.exp(self.value)
        except OverflowError:
            exp = float("inf")

        out = Node(exp, (self,))

        def backward():
            self.grad += out.value * out.grad

        out._backward = backward

        return out

    def log(self) -> "Node":
        out = Node(math.log(self.value) if self.value > 0 else float("-inf"), (self,))

        def backward():
            if self.value != 0.0:
                self.grad += ((1 / self.value)) * out.grad
            else:
                self.grad += float("inf") * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other) -> "Node":
        return self * other**-1

    def __neg__(self) -> "Node":
        return self * -1

    def __sub__(self, other) -> "Node":
        return self + (-other)

    def __rmul__(self, other) -> "Node":
        return self * other

    def __radd__(self, other) -> "Node":
        return self + other

    def __rtruediv__(self, other) -> "Node":
        return other * self**-1

    def __rsub__(self, other) -> "Node":
        return -self + other

    def __repr__(self):
        return f"Node({self.value})"

    # def __lt__(self, other) -> bool:
    #     return self.value < other.value

    # def __le__(self, other) -> bool:
    #     return self.value <= other.value

    # def __gt__(self, other) -> bool:
    #     return self.value > other.value

    # def __ge__(self, other) -> bool:
    #     return self.value >= other.value

    # def __eq__(self, other) -> bool:
    #     return self.value == other.value

    def backward(self):
        topological = []
        visited = set()

        def build_topological(node):
            if node in visited:
                return
            visited.add(node)
            for prev in node._prev:
                build_topological(prev)
            topological.append(node)

        build_topological(self)
        self.grad = 1

        for node in reversed(topological):
            node._backward()
