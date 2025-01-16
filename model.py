#!/usr/bin/env python3

import argparse
import csv
import torch
from trainer import parserow, getmodel, AKIRNN

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
    w = csv.writer(open(flags.output, "w", newline=''))
    w.writerow(("aki",))
    # check if "aki" column is present in the given dataset
    headers = next(r)
    if headers[2] == "aki":
        hidden = False
    else:
        hidden = True
    # initiate the model
    model = getmodel(False) # model does not process batch here
    model = model.to(device)
    # load trained model from file
    model.load_state_dict(torch.load("trained_model.pt", map_location=device, weights_only=True))
    model.eval()
    torch.no_grad()
    for row in r:
        age, gender, flag, times, results = parserow(row, hidden)
        x1 = torch.stack((torch.tensor(results), torch.tensor(times)), dim=1)
        x2 = torch.tensor((age, gender))
        x1 = x1.to(device)
        x2 = x2.to(device)
        output = model(x1, x2)
        predicted = torch.argmax(output)
        prediction = "n" if predicted == 0 else "y"
        w.writerow((prediction,))
    print("predictions written to "+ flags.output)

if __name__ == "__main__":
    main()