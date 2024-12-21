import torch
import torchvision


with open("data.csv") as f:
    data = f.readlines()
    data = [line.split(",") for line in data]

with open("scales.csv") as f:
    scales = f.readlines()[1:]
    scales = [list(map(float, line.split(","))) for line in scales]

x = [list(map(float, line[2:])) for line in data]
y = [int(line[1] == "M") for line in data]

# standardize features
for i in range(len(x)):
    for feature_idx in range(len(x[i])):
        mean = scales[feature_idx][0]
        std_dev = scales[feature_idx][1]
        x[i][feature_idx] = (x[i][feature_idx] - mean) / std_dev


# load model
model = torchvision.ops.MLP(30, [16, 16, 2], activation_layer=torch.nn.Tanh)
model.load_state_dict(torch.load("mlp-l2-reg.pt", weights_only=True))

# make predictions
model.eval()
predictions = model(torch.tensor(x))

# evaluate accuracy
acc = 0
for logits, yi in zip(predictions, y):
    prediction = logits.tolist()
    prediction = prediction.index(max(prediction))
    acc += prediction == yi

print(f"Accuracy: {acc / len(y) * 100:.1f}%")
