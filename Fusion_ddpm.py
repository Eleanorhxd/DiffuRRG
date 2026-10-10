
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTimestepFusion(nn.Module):
    def __init__(self, in_dim, out_dim, t_count, fusion_type='attention'):
        super(MultiTimestepFusion, self).__init__()
        self.fusion_type = fusion_type
        self.t_count = t_count

        if fusion_type == 'concat':
            self.proj = nn.Linear(in_dim * t_count, out_dim)
        elif fusion_type == 'mean':
            self.proj = nn.Linear(in_dim, out_dim)
        elif fusion_type == 'attention':
            self.att_query = nn.Linear(in_dim, in_dim)
            self.att_key = nn.Linear(in_dim, in_dim)
            self.att_value = nn.Linear(in_dim, in_dim)
            self.final_proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, features_list):
        """
        features_list: list of tensors, each shape (B, N, C)
        return: (B, N, out_dim)
        """
        if self.fusion_type == 'concat':
            feats = torch.cat(features_list, dim=-1)  # (B, N, C * t_count)
            return self.proj(feats)

        elif self.fusion_type == 'mean':
            feats = torch.stack(features_list, dim=0).mean(dim=0)  # (B, N, C)
            return self.proj(feats)

        elif self.fusion_type == 'attention':
            feats = torch.stack(features_list, dim=1)  # (B, T, N, C)
            B, T, N, C = feats.shape
            feats = feats.permute(0, 2, 1, 3).reshape(B * N, T, C)

            Q = self.att_query(feats)
            K = self.att_key(feats)
            V = self.att_value(feats)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attended = torch.matmul(attn_weights, V)

            fused = attended.mean(dim=1).view(B, N, C)
            return self.final_proj(fused)
