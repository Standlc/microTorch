import matplotlib.pyplot as plt
import nn
import random as rand
from node import Node
import torchvision
import torch

rand.seed(42)
torch.manual_seed(0)

steps = 100
batch_size = 32
lr = 2e-3
reg_lambda = 1e-2

is_torch = True


def standardize(data: list[dict[str, any]], category: str, feature_idx):
    features = [line[category][feature_idx] for line in data]

    mean = sum(features) / len(features)
    deviations = [(x - mean) ** 2 for x in features]
    variance = sum(deviations) / len(features)
    std_dev = variance**0.5

    for i in range(len(data)):
        data[i][category][feature_idx] = (features[i] - mean) / std_dev

    return mean, std_dev


def loss_fn(outputs: list[list[Node]], targets: list[int], parameters):
    if is_torch:
        acc = 0.0
        for i in range(len(outputs)):
            if outputs[i].argmax() == targets[i]:
                acc += 1

        loss = torch.functional.F.cross_entropy(outputs, targets)
        reg_loss = sum([p.pow(2).sum() for p in parameters]) * reg_lambda
        total_loss = loss + reg_loss

        return total_loss, acc / len(outputs)

    else:
        probs = [probs[target] for probs, target in zip(outputs, targets)]

        accuracy = 0.0
        for prob in probs:
            accuracy += prob.value > 0.5

        log_probs = [p.log() for p in probs]
        reg_loss = sum([p.value**2 for p in parameters]) * reg_lambda
        loss = -sum(log_probs) / len(log_probs)
        loss += reg_loss

        return loss, accuracy / len(probs)


def train(model: nn.MLP, optimizer, x: list[list[float]], y: list[int]):
    stats = []
    x_train, y_train, x_test, y_test = get_splits(x, y, 0.9)

    if is_torch:
        parameters = [p for p in model.parameters()]
    else:
        parameters = model.parameters()

    for i in range(steps):
        optimizer.zero_grad()

        # make batch
        idx = rand.sample(range(len(x_train) - 1), batch_size)
        x_batch = [x_train[i] for i in idx]
        y_batch = [y_train[i] for i in idx]

        if is_torch:
            x_batch = torch.tensor(x_batch)
            y_batch = torch.tensor(y_batch)
            outputs = model(x_batch)
        else:
            outputs = list(map(model, x_batch))

        loss, acc = loss_fn(outputs, y_batch, parameters)

        # --- test loss ---
        if is_torch:
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)
            test_outputs = model(x_test)
        else:
            test_outputs = list(map(model, x_test))

        test_loss, test_acc = loss_fn(test_outputs, y_test, parameters)
        # --- test loss ---

        loss.backward()
        optimizer.step()

        print(
            f"is_torch: {is_torch}, step {i + 1}/{steps}, loss: {loss.item():.4f}, acc: {(acc * 100):.1f}%, test loss: {test_loss.item():.4f}, test acc: {(test_acc * 100):.1f}%"
        )

        stats.append(
            {
                "train_loss": loss.item(),
                "train_acc": acc,
                "test_loss": test_loss.item(),
                "test_acc": test_acc,
            }
        )

    return stats


def plot_stats(stats):
    plt.plot([s["train_loss"] for s in stats], label="train loss")
    plt.plot([s["test_loss"] for s in stats], label="test loss")
    plt.plot([s["train_acc"] for s in stats], label="train acc")
    plt.plot([s["test_acc"] for s in stats], label="test acc")
    plt.legend(loc="upper right")
    plt.show()


def get_splits(x: list[list[float]], y: list[int], ratio: float):
    split_idx = int(len(x) * ratio)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    return x_train, y_train, x_test, y_test


def save_features_scales(scales):
    with open("scales.csv", "w") as f:
        f.write("mean,std_dev\n")
        for scale in scales:
            f.write(f"{scale[0]},{scale[1]}\n")


try:
    with open("data.csv") as f:
        raw_data = f.readlines()

except FileNotFoundError:
    print("Could not open file")

else:
    raw_lines = [line.split(",") for line in raw_data]
    rand.shuffle(raw_lines)
    x = [
        {
            "mean": [float(n) for n in line[2:12]],
            "std_err": [float(n) for n in line[12:22]],
            "worst": [float(n) for n in line[22:]],
        }
        for line in raw_lines
    ]
    y = [int(line[1] == "M") for line in raw_lines]

    data_scales = []  # mean, std_dev
    for category in ["mean", "std_err", "worst"]:
        for i in range(10):
            scales = standardize(x, category, i)
            data_scales.append(scales)

    save_features_scales(data_scales)

    if is_torch:
        model = torchvision.ops.MLP(30, [16, 16, 2], activation_layer=torch.nn.Tanh)
        print(f"number of parameters: {sum([p.numel() for p in model.parameters()])}")
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    else:
        model = nn.MLP([30, 16, 16, 2], activation=nn.TanH)
        optimizer = nn.AdamW(model.parameters(), lr)
        print(f"number of parameters: {len(model.parameters())}")
        print(model)

    features = [xi["mean"] + xi["std_err"] + xi["worst"] for xi in x]
    stats = train(model, optimizer, features, y)
    plot_stats(stats)

    if is_torch:
        torch.save(model.state_dict(), "mlp-l2-reg.pt")
    else:
        model.save("mlp-l2-reg.wei")
