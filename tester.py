#!/usr/bin/env python3

import torch
import csv
from trainer import parse, Dset, AKIRNN
from torch.utils.data import DataLoader

# import testing data
data = csv.reader(open("test.csv"))
# parse data
ages, genders, flags, testtimes, testresults = parse(data)
# load data to torch dataset (and set up dataloader)
test_dataset = Dset(ages, genders, flags, testtimes, testresults)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# initiate the model
input_size = 2  # time and result of tests
hidden_size = 5
output_size = 2  # For binary classification
layers = 1
model = AKIRNN(input_size, hidden_size, output_size, layers)
# load trained model from file
model.load_state_dict(torch.load("trained_model.pt", weights_only=True))
# evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x1, x2, labels in test_loader:
        outputs = model(x1, x2)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')