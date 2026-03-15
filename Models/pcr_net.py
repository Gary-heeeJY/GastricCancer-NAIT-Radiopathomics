import torch
import torch.nn as nn
from .abmil import ABMIL
from .gmu import GatedMultimodalUnit
from .tri_gmu import TriModalGMU  # 你写好的三模态模块


class PCRFusionNet(nn.Module):
    def __init__(self, cfg):
        super(PCRFusionNet, self).__init__()

        self.use_ct = cfg.get('use_ct', False)  # 默认 False

        self.visual_dim = cfg.visual_dim
        self.text_dim = cfg.text_dim
        self.hidden_dim = cfg.hidden_dim
        self.dropout_rate = cfg.dropout_rate

        self.mil = ABMIL(input_dim=self.visual_dim, hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)

        # 根据开关动态选择融合模块
        if self.use_ct:
            self.ct_dim = cfg.ct_dim
            self.fusion = TriModalGMU(
                visual_dim=self.visual_dim,
                text_dim=self.text_dim,
                ct_dim=self.ct_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            )
        else:
            self.fusion = GatedMultimodalUnit(
                visual_dim=self.visual_dim,
                text_dim=self.text_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, patch_features, text_features, ct_features=None):
        patient_visual_feat, attention_scores = self.mil(patch_features)

        # 动态传参融合
        if self.use_ct and ct_features is not None:
            fused_feat = self.fusion(patient_visual_feat, text_features, ct_features)
        else:
            fused_feat = self.fusion(patient_visual_feat, text_features)

        logits = self.classifier(fused_feat)
        return logits, attention_scores