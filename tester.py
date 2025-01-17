#!/usr/bin/env python3

import torch
import csv
from trainer import parse, getmodel
from torch.utils.data import DataLoader

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")
# import testing data
data = csv.reader(open("test.csv"))
# parse and load data
test_loader, _, _ = parse(data, device)
# get model structure
model = getmodel(True) # model processes batch here
model = model.to(device)
# load trained model from file
model.load_state_dict(torch.load("trained_model.pt", map_location=device, weights_only=True))
# evaluate model
model.eval()
total = 0
correct = 0
truepositive = 0
falsepositive = 0
falsenegative = 0
print("evaluation started")
with torch.no_grad():
    for x1, x2, labels in test_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        outputs = model(x1, x2)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        for i in range(labels.size(0)):
            if predicted[i] == labels[i]:
                correct += 1
                if labels[i] == 1:
                    truepositive += 1
            else:
                if predicted[i] == 1:
                    falsepositive += 1
                else:
                    falsenegative += 1
accuracy = 100 * correct / total  # accuracy
f3 = 10 * truepositive / (10 * truepositive + falsepositive + 9 * falsenegative)  # f3 score
print(f'Accuracy: {accuracy:.2f}%')
print(f'f3 score: {f3:.2f}')
