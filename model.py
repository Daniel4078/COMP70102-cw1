#!/usr/bin/env python3

import argparse
import csv
import torch
from trainer import parse, getmodel, AKIRNN

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
    # parse and load data
    test_loader, _, _ = parse(r, device, False)
    # initiate the model
    model = getmodel()
    model = model.to(device)
    # load trained model from file
    model.load_state_dict(torch.load("trained_model.pt", map_location=device, weights_only=True))
    model.eval()
    print("prediction started")
    prediction = []
    with torch.no_grad():
        for x1_padded, x2, lengths, labels in test_loader:
            outputs = torch.sigmoid(model(x1_padded, x2, lengths))
            for i in range(labels.size(0)):
                prediction.append("n" if outputs[i] < 0.5 else "y")
    for p in prediction:
        w.writerow((p,))
    print("predictions written to " + flags.output)

if __name__ == "__main__":
    main()