import numpy as np
import matplotlib.pyplot as plt
from nn import MLP
from optimizer import Optimizer
import random as rand
from node import Node

steps = 50
batch_size = 32
lr = 0.01


def standardize(data: list[dict[str, any]], category: str, feature_idx) -> None:
    features = [line[category][feature_idx] for line in data]

    mean = sum(features) / len(features)
    deviations = [(x - mean) ** 2 for x in features]
    variance = sum(deviations) / len(features)
    std_dev = variance**0.5

    for i in range(len(features)):
        data[i][category][feature_idx] = (features[i] - mean) / std_dev


def loss_fn(outputs: list[list[Node]], targets: list[int]):
    accuracy = 0.0

    probs = [probs[target] for probs, target in zip(outputs, targets)]

    for prob in probs:
        if prob.value > 0.5:
            accuracy += 1

    log_probs = [p.log() for p in probs]
    loss = -sum(log_probs) / len(log_probs)

    return loss, accuracy / len(probs)


def train(model: MLP, optimizer, x: list[list[float]], y: list[int]):
    for i in range(steps):

        # make batch
        idx = [rand.randint(0, len(x) - 1) for _ in range(batch_size)]
        x_batch = [x[i] for i in idx]
        y_batch = [y[i] for i in idx]

        outputs = list(map(model, x_batch))
        loss, acc = loss_fn(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {i + 1}/{steps}, loss: {loss.value:.4f}, accuracy: {acc * 100}%")


def get_splits(x: list[dict[str, any]], y: list[int], ratio: float):
    split_idx = int(len(x) * ratio)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    return x_train, y_train, x_test, y_test


try:
    with open("data.csv") as f:
        raw_data = f.read()

except FileNotFoundError:
    print("Could not open file")

else:
    raw_lines = [line.split(",") for line in raw_data.split("\n")][:-1]
    x = [
        {
            "mean": [float(n) for n in line[2:12]],
            "std_err": [float(n) for n in line[12:22]],
            "worst": [float(n) for n in line[22:]],
        }
        for line in raw_lines
    ]
    y = [(0 if line[1] == "B" else 1) for line in raw_lines]

    for category in ["mean", "std_err", "worst"]:
        for i in range(10):
            standardize(x, category, i)

    model = MLP([10, 24, 24, 24, 2])
    print(f"number of parameters: {len(model.parameters())}")
    optimizer = Optimizer(model.parameters(), lr)

    x_train, y_train, x_test, y_test = get_splits(x, y, 0.9)
    train(model, optimizer, [xi["mean"] for xi in x_train], y_train)
