#!/usr/bin/env python3

import argparse
import csv
import random
import torch
from trainer import parserow, AKIRNN

def main():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    r = csv.reader(open(flags.input))
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    next(r) # skip headers
    # initiate the model
    input_size = 2  # time and result of tests
    hidden_size = 5
    output_size = 2  # For binary classification
    layers = 1
    model = AKIRNN(input_size, hidden_size, output_size, layers)
    # load trained model from file
    model.load_state_dict(torch.load("trained_model.pt", map_location=device, weights_only=True))
    model.eval()
    torch.no_grad()
    for row in r:
        age, gender, flag, times, results = parserow(row)
        x1 = torch.stack((torch.tensor(results), torch.tensor(times)), dim=1)
        x2 = torch.tensor((age, gender))
        output = model(x1, x2)
        _, predicted = torch.max(output)
        w.writerow(("n" if predicted == 0 else "y",))

if __name__ == "__main__":
    main()