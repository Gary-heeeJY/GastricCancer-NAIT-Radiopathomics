import torch
import torch.nn as nn


class CTOnlyNet(nn.Module):
    def __init__(self, cfg):
        super(CTOnlyNet, self).__init__()

        self.ct_dim = cfg.ct_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout_rate = cfg.dropout_rate

        # 将 BatchNorm1d 替换为 LayerNorm，完美兼容 batch_size=1
        self.classifier = nn.Sequential(
            nn.Linear(self.ct_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # 👈 修改这里
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim, 32),
            nn.LayerNorm(32),  # 👈 修改这里
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, ct_features):
        # ct_features shape: [1, N_features]
        logits = self.classifier(ct_features)

        # 返回第二个参数为 None，是为了和多模态的 API 保持一致 (兼容外部的解包)
        return logits, None