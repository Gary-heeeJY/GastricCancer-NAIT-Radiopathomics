import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultimodalUnit(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, dropout_rate=0.3):
        super(GatedMultimodalUnit, self).__init__()

        # 降维/升维，对齐到公共隐空间
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.txt_proj = nn.Linear(text_dim, hidden_dim)

        # 门控计算层
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v_feat, t_feat):
        """
        v_feat: [1, visual_dim]
        t_feat: [1, text_dim]
        """
        # 投影并激活
        h_v = F.relu(self.vis_proj(v_feat))
        h_t = F.relu(self.txt_proj(t_feat))

        # 拼接并计算门控系数 z (取值 0~1 之间)
        concat_feat = torch.cat([h_v, h_t], dim=-1)
        z = torch.sigmoid(self.gate_layer(concat_feat))

        # 加权融合
        fused_feat = z * h_v + (1 - z) * h_t
        fused_feat = self.dropout(fused_feat)

        return fused_feat