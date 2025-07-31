import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from open_data import open_data_dir
from geodataclass import FlattenedData
from open_fire_data import FlattenedTruthTable

class DNN(nn.Module):
    '''
    predictor for 1 variable
    '''

    def __init__(self, input_size = 10, output_size = 1):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )

    def forward(self, x):
        return self.network(x)
    

class geoDataset(Dataset):

    def __init__(self, input_data: FlattenedData, fire_data: FlattenedTruthTable):
        '''
        input data is climate variables <br>
        training data is fire data
        '''

        self.input_data = input_data
        self.fire_data = fire_data

        ## validation eg have the same number of datapoints


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        '''
        should return (features, label), where
        - features are relevent input datapoints
        - label is the relevant fire data (either 1 or 0)
        '''

        features = [self.input_data[index, i] for i in range(self.input_data.feature_num)]
        label = self.training_data[index]

        return features, label


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = DNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    batch_size = 64

    # Create data loaders.
    input_data = FlattenedData(
        data=open_data_dir('./data/alaska_TEST_DATA'),
        prior_data = open_data_dir('./data/alaska_prior')
    )

    fire_data = FlattenedTruthTable(
        pd.read_csv(
            './data/FIRE/alaska_range_csv/data.csv',
            parse_dates = ['acq_date']
        )
    )
    print('data  read')

    training_data = geoDataset(input_data, fire_data)
    test_data = training_data

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")