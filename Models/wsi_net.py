import torch
import torch.nn as nn
from .abmil import ABMIL


class WSIOnlyNet(nn.Module):
    def __init__(self, cfg):
        super(WSIOnlyNet, self).__init__()
        self.visual_dim = cfg.visual_dim  # 1024
        self.hidden_dim = cfg.hidden_dim  # 256
        self.dropout_rate = cfg.dropout_rate

        self.mil = ABMIL(input_dim=self.visual_dim, hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)

        # 🎯 修复点：ABMIL 输出依然是 1024 维，所以这里必须用 nn.Linear(1024, 256) 降维
        self.classifier = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),  # 1024 -> 256
            nn.LayerNorm(self.hidden_dim),  # 层归一化，稳定训练
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim, 32),  # 256 -> 32
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, patch_features):
        # patient_visual_feat 的形状是 [1, 1024]
        patient_visual_feat, attention_scores = self.mil(patch_features)
        logits = self.classifier(patient_visual_feat)
        return logits, attention_scores