# models.py

import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out, _ = self.rnn(x)
        out = out.contiguous().view(batch_size * seq_len, -1)
        out = self.fc(out)
        return out
