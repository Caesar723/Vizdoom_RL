import torch
import torch.nn as nn



class GameTransformer(nn.Module):
    def __init__(self,d_t):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_t, nhead=1,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):#传输几帧数之前到目前帧的数据
        return self.transformer(x)
    

if __name__ == "__main__":
    transformer = GameTransformer(d_t=128)
    x = torch.randn(1, 10, 128)
    print(transformer(x).shape)