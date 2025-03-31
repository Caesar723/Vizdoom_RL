import torch
import torch.nn as nn



class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: [B, N, H]
        attn_weights = self.attn_fc(x).squeeze(-1)  # [B, N]
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B, N]
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled
