#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import time
import statistics
from datetime import datetime, timedelta
import csv


# define a GRU/RNN model
class AKIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):
        super(AKIRNN, self).__init__()
        self.l = layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size + 2, output_size)

    def forward(self, x1, x2):
        # Initialize hidden states of RNN with zeros
        h0 = torch.zeros(self.l, x1.size(0), hidden_size)
        out, _ = self.rnn(x1, h0)
        combined = torch.cat((out[:, -1, :], x2), dim=1)
        final = self.fc(combined)
        return final


# a dataset class that pad the sequences into same length for batch training
class Dset(Dataset):
    def __init__(self, age, gender, flag, date, result):
        self.age = age
        self.gender = gender
        self.flag = flag
        self.length = len(self.age)
        # padding the sequences
        self.date = pad_sequence(date, batch_first=True)
        self.result = pad_sequence(result, batch_first=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x1 = torch.stack((self.result[idx], self.date[idx]), dim=1)
        x2 = torch.tensor((self.age[idx], self.gender[idx]))
        label = torch.tensor(self.flag[idx])
        return x1, x2, label


# function to parse data, used in other files
def parse(data):
    next(data)
    ages = []
    genders = []
    flags = []
    alltimes = []
    allresults = []
    for row in data:
        age, gender, flag, times, results = parserow(row)
        ages.append(age)
        genders.append(gender)
        flags.append(flag)
        alltimes.append(torch.tensor(times))
        allresults.append(torch.tensor(results))
    return ages, genders, flags, alltimes, allresults


def parserow(row):
    length = (len(row) - 3) // 2
    age = int(row[0])
    gender = 1 if row[1] == "f" else 2
    if row[2] == "":
        flag = -1
    else:
        flag = 0 if row[2] == "n" else 1
    times = []
    results = []
    started = False
    prev = 0
    for i in range(length - 1, -1, -1):
        if row[3 + i * 2] == "":
            continue
        if not started:
            started = True
            prev = datetime.fromisoformat(row[3 + i * 2])
            times.append(1)
            results.append(float(row[4 + i * 2]))
        else:
            current = datetime.fromisoformat(row[3 + i * 2])
            times.append((prev - current) / timedelta(seconds=1) + 1)
            results.append(float(row[4 + i * 2]))
            prev = current
    return age, gender, flag, times, results


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
    # import training data
    data = csv.reader(open("training.csv"))
    # parse data
    ages, genders, flags, testtimes, testresults = parse(data)
    # load data to torch dataset (and set up dataloader)
    train_dataset = Dset(ages, genders, flags, testtimes, testresults)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # initiate the model
    input_size = 2  # time and result of tests
    hidden_size = 5
    output_size = 2  # For binary classification
    layers = 1
    model = AKIRNN(input_size, hidden_size, output_size, layers)
    # train the model
    print("training started")
    start = time.time()
    n_epoch = 50
    report_every = 5
    learning_rate = 0.15
    criterion = nn.CrossEntropyLoss() # TODO: change loss to f3 score
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for iter in range(1, n_epoch + 1):
        model.zero_grad()
        e_loss = []
        for x1, x2, label in train_loader:
            outputs = model(x1, x2)
            loss = criterion(outputs, label)
            e_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {statistics.mean(e_loss):.4f}")
    end = time.time()
    print(f"training took {end - start}s")
    # save the trained model
    torch.save(model.state_dict(), "trained_model.pt")
    print("trained model saved as trained_model.pt")
