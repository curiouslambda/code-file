# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        if self.training and x.size(0) > 1:  # 학습 모드일 때만 배치 정규화 적용
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:  # 평가 모드일 때는 배치 정규화 생략
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        return self.fc3(x)
