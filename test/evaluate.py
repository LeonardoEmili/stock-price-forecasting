#!/usr/bin/env python3

import torch
import torch.nn as nn
import dataclasses
import numpy as np
import sys
import getopt
from random import sample

samples_num = 10
try:
    # Parse the options
    opts, _ = getopt.getopt(sys.argv[1:], "n:")
    samples_num = int(opts[0][1])
except IndexError: pass
except Exception:
    print('usage: evaluate.py [-n N]', file=sys.stderr)
    exit(1)

class SimpleLSTM(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lstm = nn.LSTM(hparams.input_dim, hparams.hidden_dim,
                            num_layers=hparams.num_layers, batch_first=True)
        self.fc1 = nn.Linear(hparams.hidden_dim, hparams.hidden_dim//2)
        self.fc2 = nn.Linear(hparams.hidden_dim//2, 1)
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x: np.ndarray):
        x, (h, c) = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1]))
        x = self.fc2(x)
        return x.squeeze()


@dataclasses.dataclass
class HParams:
    input_dim: int = 35
    window_size: int = 10
    batch_size: int = 1024
    hidden_dim: int = 128
    num_layers: int = 1
    lr: int = 0.01
    momentum: int = 0.3
    dropout: int = 0.0


def r2_score(y_hat: np.ndarray, y: np.ndarray) -> float:
    ''' Computes the R Squared coefficient between y_hat and y. '''
    rss = torch.sum((y - y_hat) ** 2)
    tss = torch.sum((y - y.mean()) ** 2)
    return 1 - rss / tss


def adjusted_r2_score(y_hat: np.ndarray, y: np.ndarray, p: int) -> float:
    ''' Computes the Adjusted R Squared coefficient between y_hat and y. '''
    rss = torch.sum((y - y_hat) ** 2)
    tss = torch.sum((y - y.mean()) ** 2)
    df_e = y.shape[0] - p - 1
    df_t = y.shape[0] - 1
    adj_coef = df_t/df_e if df_t/df_e > 0 else 1  # sample size < features
    return 1 - (rss / tss) * adj_coef


@torch.no_grad()
def evaluate(
        model,
        loss_fn,
        data,
        model_name='LSTM',
        log_precision=4):
    x, y = data['X'], data['Y']
    y_hat = model(x.float())
    loss = loss_fn(y_hat, y.float())
    r2 = r2_score(y_hat, y)
    adj_r2 = adjusted_r2_score(y_hat, y, x.shape[1] * x.shape[2])

    print('===================================')
    print(f'Performances of the {model_name} over {x.shape[0]} samples\n')
    print(f'MSE:\t\t{loss}\nR2:\t\t{r2}\nAdjusted R2:\t{adj_r2}')
    print('===================================')

    pairs = list(zip(y, y_hat))
    samples = sample(pairs, samples_num)

    print(f'Log {samples_num} samples\n')
    print('Real\tPredicted\tDifference')
    for sample_y, sample_y_hat in samples:
        diff = round(sample_y.item() - sample_y_hat.item(), log_precision)
        diff = f'+{diff}' if diff > 0 else diff
        real = round(sample_y.item(), log_precision)
        pred = round(sample_y_hat.item(), log_precision)
        print(f'{real}\t{pred}\t\t{diff}')
    print('===================================')


if __name__ == '__main__':
    # Initialize the model and load pretrained parameters
    hparams = HParams()
    model = SimpleLSTM(hparams)
    model.load_state_dict(torch.load('weights.pt'))
    model.eval()

    # Define the objective loss function
    loss_fn = nn.MSELoss()

    # Load samples from test set
    test_data = torch.load('test_data.pt')

    evaluate(model, loss_fn, test_data)
