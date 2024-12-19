import random
import nn

# load data
with open("data.csv", "r") as f:
    data = f.readlines()
    data = random.sample(data, 100)

# load mean and std. dev. of data features
with open("scales.csv", "r") as f:
    scales = f.readlines()[1:]

# split in x, y
data = [line.split(",") for line in data]
x = [list(map(float, line[2:])) for line in data]
y = [int(line[1] == "M") for line in data]

# standardize features
scales = [list(map(float, line.split(","))) for line in scales]
for i in range(len(x)):
    for feature_idx in range(len(x[i])):
        mean = scales[feature_idx][0]
        std_dev = scales[feature_idx][1]
        x[i][feature_idx] = (x[i][feature_idx] - mean) / std_dev


# load model weights
model = nn.MLP([30, 16, 16, 2], activation=nn.TanH())
model.load_state_dict("mlp.wei")
print(f"number of parameters: {len(model.parameters())}")

# make predictions
predictions = list(map(model, x))
acc = 0
for logits, yi in zip(predictions, y):
    logits = [l.value for l in logits]
    prediction = logits.index(max(logits))
    acc += prediction == yi

# print accuracy
print(f"Accuracy: {acc / len(y) * 100:.1f}%")
