#!/usr/bin/env python3

import torch  # used to build and train the nn model
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import time
import statistics
from datetime import datetime, timedelta
import csv


# define a LSTM model
class AKIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mid_size, layers):
        super(AKIRNN, self).__init__()
        self.l = layers
        self.hidden = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 2, mid_size)
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(mid_size, output_size)

    def forward(self, x1_padded, x2, lengths):
        x1_pack = pack_padded_sequence(x1_padded, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(x1_pack)
        ends = h_n[-1]
        combined = torch.cat([ends, x2], 1)
        final = self.fc1(self.act(self.fc(combined)))
        return final


# initiate the model
def getmodel():
    input_size = 2  # time and result of tests
    hidden_size = 32
    output_size = 1  # For binary classification
    layers = 1
    mid_size = 32
    model = AKIRNN(input_size, hidden_size, output_size, mid_size, layers)
    return model


# a dataset class that pad the sequences into same length for batch training
class Dset(Dataset):
    def __init__(self, age, gender, flag, date, result):
        self.age = age
        self.gender = gender
        self.flag = flag
        self.length = len(self.age)
        self.date = date
        self.result = result

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x1 = torch.stack((self.result[idx], self.date[idx]), dim=1)
        x2 = torch.tensor((self.age[idx], self.gender[idx]))
        label = torch.tensor(self.flag[idx], dtype=torch.float32)
        return (x1, x2, label)


class PadSequence:
    def __call__(self, batch):  # each element in "batch" is a tuple (x1, x2, label).
        # Get each sequence of data and label, pad x1
        x1, x2, labels = zip(*batch)
        x2 = torch.stack(x2)
        labels = torch.stack(labels)
        # store the length of each sequence
        lengths = torch.LongTensor([len(x) for x in x1])
        # pad x1
        x1_padded = pad_sequence(x1, batch_first=True)
        return x1_padded, x2, lengths, labels


# function to parse data and load them in a dataloader
def parse(data, device, not_inference):
    # check if "aki" column is present in the given dataset
    headers = next(data)
    if headers[2] == "aki":
        hidden = False
    else:
        hidden = True
    ages = []
    genders = []
    flags = []
    alltimes = []
    allresults = []
    positive = 0
    negative = 0
    for row in data:
        age, gender, flag, times, results = parserow(row, hidden)
        if flag == 0:
            negative += 1
        elif flag == 1:
            positive += 1
        ages.append(age)
        genders.append(gender)
        flags.append(flag)
        alltimes.append(torch.tensor(times))
        allresults.append(torch.tensor(results))
    # load data to torch dataset (and set up dataloader)
    train_dataset = Dset(ages, genders, flags, alltimes, allresults)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=not_inference,
                              generator=torch.Generator(device=device), collate_fn=PadSequence())
    return train_loader, negative, positive


def parserow(row, hidden):
    if hidden:
        length = (len(row) - 2) // 2
    else:
        length = (len(row) - 3) // 2
    age = int(row[0])
    gender = 1 if row[1] == "f" else 0
    if hidden:
        flag = -1
    else:
        flag = 0 if row[2] == "n" else 1
    times = []
    results = []
    started = False
    prev = 0
    # convert test date info to the time difference (hours) from that test to most recent test
    # assuming the test data are already sorted by time
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
            times.append((prev - current) / timedelta(seconds=1) / 3600 + 1)
            results.append(float(row[offset + 1]))
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
    train_loader, n, p = parse(data, device, True)
    # use invert class sampling count to balance the loss
    weight = torch.tensor(n / p, dtype=torch.float32)
    print(f"number of negatives:{n}, number of positives:{p}")
    # get model structure
    model = getmodel()
    model.to(device)
    # train the model
    print("training started")
    start = time.time()
    n_epoch = 200
    report_every = 10
    learning_rate = 0.0005
    weight_decay = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=3 * weight) # put more focus on positive cases
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
    for iter in range(1, n_epoch + 1):
        model.zero_grad()
        e_loss = []
        for x1_padded, x2, lengths, labels in train_loader:
            labels = labels.view(-1, 1)
            outputs = model(x1_padded, x2, lengths)
            loss = criterion(outputs, labels)
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
