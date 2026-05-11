"""
Model definitions for Mosquito Flight Trajectory prediction.
"""
import math
import torch
import torch.nn as nn


class MosquitoGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout_rate=0.2):
        super(MosquitoGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 레이어 (num_layers > 1 일때 레이어 사이에 dropout 적용)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length=11, input_size=3)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out


class WingLoss(nn.Module):
    def __init__(self, w=0.05, epsilon=0.01):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, y_pred, y_true):
        x = torch.abs(y_pred - y_true)
        loss = torch.where(
            x < self.w,
            self.w * torch.log(1.0 + x / self.epsilon),
            x - self.c
        )
        return loss.mean()
