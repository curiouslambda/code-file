import torch
import torch.nn as nn
from config import Config

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.hidden_size = Config.hidden_size
        self.rnn = nn.RNN(Config.input_size, self.hidden_size, batch_first=True)
        self.fc_category = nn.Linear(self.hidden_size, Config.num_categories)  # 카테고리 선택
        self.fc_probability = nn.Linear(self.hidden_size, 1)  # 승리할 확률

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # 마지막 시퀀스 출력
        category_output = self.fc_category(out)
        probability_output = self.fc_probability(out)
        return category_output, probability_output
