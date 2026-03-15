import torch
import torch.nn as nn
import torch.nn.functional as F


class TriModalGMU(nn.Module):
    def __init__(self, visual_dim, text_dim, ct_dim, hidden_dim, dropout_rate=0.5):
        super(TriModalGMU, self).__init__()

        # 🛡️ 强化点：在降维的同时加入 LayerNorm，强制对齐三个模态的分布差异
        self.vis_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.ct_proj = nn.Sequential(
            nn.Linear(ct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 联合计算三者的门控权重 (Gate)
        self.gate_layer = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v_feat, t_feat, c_feat):
        # 激活投影特征
        h_v = F.relu(self.vis_proj(v_feat))
        h_t = F.relu(self.txt_proj(t_feat))
        h_c = F.relu(self.ct_proj(c_feat))

        # 拼接并生成独立权重
        concat_feat = torch.cat([h_v, h_t, h_c], dim=-1)
        gates = torch.sigmoid(self.gate_layer(concat_feat))

        # 将 Gates 分割给三个模态
        z_v, z_t, z_c = torch.split(gates, h_v.size(-1), dim=-1)

        # Softmax 归一化权重
        z_stack = torch.stack([z_v, z_t, z_c], dim=0)
        z_norm = F.softmax(z_stack, dim=0)
        z_v, z_t, z_c = z_norm[0], z_norm[1], z_norm[2]

        # 最终加权融合
        fused_feat = z_v * h_v + z_t * h_t + z_c * h_c
        return self.dropout(fused_feat)