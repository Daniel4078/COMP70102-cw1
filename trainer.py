#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import time
import statistics
from datetime import datetime, timedelta
import csv


# define a LSTM model
class AKIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, batched=True):
        super(AKIRNN, self).__init__()
        self.l = layers
        self.b = batched
        self.hidden = hidden_size
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(2 * hidden_size + 2, output_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 2, output_size)

    def forward(self, x1, x2):
        out, _ = self.rnn(x1)
        # check if given data is batched
        if not self.b:
            combined = torch.cat((out[-1, :], x2))
        else:
            combined = torch.cat((out[:, -1, :], x2), dim=1)
        final = self.fc(combined)
        return final


# initiate the model
def getmodel(batched):
    input_size = 2  # time and result of tests
    hidden_size = 20
    output_size = 2  # For binary classification
    layers = 2
    model = AKIRNN(input_size, hidden_size, output_size, layers, batched)
    return model


# a dataset class that pad the sequences into same length for batch training
class Dset(Dataset):
    def __init__(self, age, gender, flag, date, result):
        self.age = age
        self.gender = gender
        self.flag = flag
        self.length = len(self.age)
        # padding the sequences
        self.date = pad_sequence(date, batch_first=True, padding_value=-1, padding_side='left')
        self.result = pad_sequence(result, batch_first=True, padding_value=-1, padding_side='left')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x1 = torch.stack((self.result[idx], self.date[idx]), dim=1)
        x2 = torch.tensor((self.age[idx], self.gender[idx]))
        label = torch.tensor(self.flag[idx])
        return x1, x2, label


# function to parse data and load them in a dataloader
def parse(data, device):
    next(data)
    ages = []
    genders = []
    flags = []
    alltimes = []
    allresults = []
    positive = 0
    negative = 0
    for row in data:
        age, gender, flag, times, results = parserow(row, False)
        if age == 0:
            negative += 1
        elif age == 1:
            positive += 1
        ages.append(age)
        genders.append(gender)
        flags.append(flag)
        alltimes.append(torch.tensor(times))
        allresults.append(torch.tensor(results))
    # load data to torch dataset (and set up dataloader)
    train_dataset = Dset(ages, genders, flags, alltimes, allresults)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    return train_loader, negative, positive


def parserow(row, hidden):
    if hidden:
        length = (len(row) - 2) // 2
    else:
        length = (len(row) - 3) // 2
    age = int(row[0])
    gender = 1 if row[1] == "f" else 2
    if hidden:
        flag = -1
    else:
        flag = 0 if row[2] == "n" else 1
    times = []
    results = []
    started = False
    prev = 0
    for i in range(length - 1, -1, -1):
        offset = (2 + i * 2) if hidden else (3 + i * 2)
        if row[offset] == "":
            continue
        if not started:
            started = True
            prev = datetime.fromisoformat(row[offset])
            times.append(1)
            results.append(float(row[offset + 1]))
        else:
            current = datetime.fromisoformat(row[offset])
            times.append((prev - current) / timedelta(seconds=1) + 1)
            results.append(float(row[offset + 1]))
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
    # parse and load data
    train_loader, n, p = parse(data, device)
    # use invert class sampling count to balance the loss
    weight = torch.tensor([p/(n+p), n/(n+p)], dtype=torch.float32)
    print(f"number of negatives:{n}, number of positives:{p}")
    # get model structure
    model = getmodel(True) # model processes batch here
    model.to(device)
    # train the model
    print("training started")
    start = time.time()
    n_epoch = 300  # 300 takes about 1200 secs = 20 mins to train
    report_every = 10
    learning_rate = 0.001
    weight_decay = 0
    criterion = nn.CrossEntropyLoss(weight=weight)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
    for iter in range(1, n_epoch + 1):
        model.zero_grad()
        e_loss = []
        for x1, x2, label in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)
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
    torch.save(model.state_dict(), "trained_model.pt")
    print("current best trained model saved as trained_model.pt")
