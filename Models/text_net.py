import torch
import torch.nn as nn


class TextOnlyNet(nn.Module):
    def __init__(self, cfg):
        super(TextOnlyNet, self).__init__()
        self.text_dim = cfg.text_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout_rate = cfg.dropout_rate

        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, text_features):
        logits = self.classifier(text_features)
        return logits, None