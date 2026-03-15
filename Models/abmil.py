import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    # 确保加入了 dropout_rate 参数
    def __init__(self, input_dim=1024, hidden_dim=64, dropout_rate=0.6):
        super(ABMIL, self).__init__()
        self.attention_a = nn.Linear(input_dim, hidden_dim)
        self.attention_b = nn.Linear(input_dim, hidden_dim)
        self.attention_c = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(torch.tanh(a) * torch.sigmoid(b))
        A = torch.transpose(A, 1, 0)
        A = torch.nn.functional.softmax(A, dim=1)

        x = self.dropout(x)  # 在聚合前应用 dropout
        patient_feat = torch.mm(A, x)
        return patient_feat, A