import torch
import torch.nn as nn
from .abmil import ABMIL
from .gmu import GatedMultimodalUnit


class WSICTFusionNet(nn.Module):
    def __init__(self, cfg):
        super(WSICTFusionNet, self).__init__()
        self.visual_dim = cfg.visual_dim  # 1024
        self.ct_dim = cfg.ct_dim  # 107
        self.hidden_dim = cfg.hidden_dim  # 256
        self.dropout_rate = cfg.dropout_rate

        self.mil = ABMIL(input_dim=self.visual_dim, hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)

        # 🎯 修复点：传入 GMU 的 visual_dim 必须是 1024
        # GMU 会自动把 1024 维病理和 107 维 CT 都映射到 256 维 (hidden_dim) 再进行融合
        self.fusion = GatedMultimodalUnit(
            visual_dim=self.visual_dim,
            text_dim=self.ct_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )

        # GMU 融合后的输出固定是 hidden_dim (256维)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),  # 256 -> 32
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, patch_features, ct_features):
        # 1. 提取病理特征 [1, 1024]
        patient_visual_feat, attention_scores = self.mil(patch_features)

        # 2. 传入 GMU 门控融合
        # v_feat=病理(1024), t_feat=CT(107) -> 输出融合特征 [1, 256]
        fused_feat = self.fusion(patient_visual_feat, ct_features)

        # 3. 分类预测
        logits = self.classifier(fused_feat)
        return logits, attention_scores