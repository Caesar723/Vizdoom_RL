import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdd(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, fx):
        res = self.proj(x) if self.proj else x
        return self.norm(res + fx)
    

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        print(dim_Q, dim_K, dim_V, num_heads)
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        B, Nq, _ = Q.size()
        Nk = K.size(1)
        d = self.fc_q.out_features // self.num_heads
        # print(K.shape)
        # print(self.fc_k(K).shape)
        K_forK = self.fc_k(K)
        K_forV = self.fc_v(K)
        

        Q = self.fc_q(Q).view(B, Nq, self.num_heads, d).transpose(1, 2)  # [B, H, Nq, d]
        K = K_forK.view(B, Nk, self.num_heads, d).transpose(1, 2)  # [B, H, Nk, d]
        # print(K.shape)
        V = K_forV.view(B, Nk, self.num_heads, d).transpose(1, 2)  # [B, H, Nk, d]

        scores = Q @ K.transpose(-1, -2) / d**0.5   
        if mask is not None:
            mask = mask[:, None, None, :]  # [B, 1, 1, Nk]
            # print(scores)
            scores = scores.masked_fill(~mask, float('-inf'))
            # print(scores)
        A = torch.softmax(scores, dim=-1)      # [B, H, Nq, Nk]
        # print(A)
        O = (A @ V).transpose(1, 2).contiguous().view(B, Nq, -1)         # [B, Nq, dim_V]
        return self.fc_o(O)
    
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out * 2, dim_out)
        )
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

        self.residual_add1 = ResidualAdd(dim_in, dim_out)
        self.residual_add2 = ResidualAdd(dim_out, dim_out)
    def forward(self, X, mask=None):
        A = self.mab(X, X, mask)
        X = self.residual_add1(X, A)
        F = self.ffn(X)
        X = self.residual_add2(X, F)
        return X

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads)
        self.ln = nn.LayerNorm(dim)

    def forward(self, X, mask=None):
        B = X.size(0)
        seed = self.seed_vectors.expand(B, -1, -1)
        return self.ln(self.mab(seed, X, mask))

class SetTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_outputs=1):
        super().__init__()
        self.SAB1 = SAB(input_dim, embed_dim, num_heads)
        self.SAB2 = SAB(embed_dim, embed_dim, num_heads)
        self.PMA = PMA(embed_dim, num_heads, num_outputs)

    def forward(self, X , mask=None):  # [B, N, D]
        X = self.SAB1(X, mask)
        X = self.SAB2(X, mask)
        return self.PMA(X, mask).squeeze(1)  # [B, D]
    

if __name__ == "__main__":
    #while True:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    Ns = [10, 5, 3, 1]
    batch = [torch.randn(n, 100) for n in Ns]
    print([batch[i].shape for i in range(len(batch))])
    padded = pad_sequence(batch, batch_first=True)  # [B, N_max, D]
    mask = (torch.arange(padded.shape[1])[None, :] < torch.tensor(Ns)[:, None]).detach().numpy()  # [B, N_max]
    print(mask)
    print(padded.shape)
    print(mask.shape)
    data = padded
    mask = mask
    model = SetTransformer(100)

    mask = torch.tensor(mask)
    output = model(data, mask)
    # print(output)
    print(output.shape)
    