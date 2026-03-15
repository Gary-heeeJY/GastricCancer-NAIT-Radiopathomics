import torch
import torch.nn as nn
from .gmu import GatedMultimodalUnit


class TextCTFusionNet(nn.Module):
    def __init__(self, cfg):
        super(TextCTFusionNet, self).__init__()
        self.text_dim = cfg.text_dim  # 1024
        self.ct_dim = cfg.ct_dim  # 107
        self.hidden_dim = cfg.hidden_dim  # 256
        self.dropout_rate = cfg.dropout_rate

        # 复用 GMU 结构：将 visual_dim 坑位当做文本传入，text_dim 坑位当做 CT 传入
        self.fusion = GatedMultimodalUnit(
            visual_dim=self.text_dim,
            text_dim=self.ct_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )

        # 融合后固定输出 hidden_dim (256维)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(32, 1)
        )

    def forward(self, text_features, ct_features):
        # 1. 传入 GMU 门控融合
        # v_feat=文本(1024), t_feat=CT(107) -> 输出融合特征 [1, 256]
        fused_feat = self.fusion(text_features, ct_features)

        # 2. 分类预测
        logits = self.classifier(fused_feat)

        # 保持统一的返回值格式 (文本和CT没有 attention 权重，所以返回 None)
        return logits, None