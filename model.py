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
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out


class MosquitoGRU_M2M(nn.Module):
    """GRU predicting both +40ms and +80ms future positions.

    Output: (B, 6) = [pred_40ms (3D), pred_80ms (3D)].
    For inference/submission use the last 3 values (pred_80ms).
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_40 = nn.Linear(hidden_size, 3)
        self.fc_80 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x: (B, T, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        return torch.cat([self.fc_40(out), self.fc_80(out)], dim=1)  # (B, 6)


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
