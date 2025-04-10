import torch
import torch.nn as nn
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.setTransfer import SetTransformer
from layers.lstm import GRU
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class NormalNet(nn.Module):
    def __init__(self,hidden_size=128,image_size=84):
        super().__init__()
        self.hidden_size=hidden_size

        #self.lstm=GRU(hidden_size,hidden_size)

        # self.state_encoder=nn.Sequential(
        #     nn.Linear(2, hidden_size),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_size)
        # )

        # self.label_mlp=nn.Sequential(
        #     nn.Linear(label_size,hidden_size),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_size)
        # )
        #self.label_norm = nn.LayerNorm(hidden_size)
        self.conv1=self.generate_conv2d_layer(1)
        self.conv2=self.generate_conv2d_layer(4)


        self.gru=GRU(image_size//8*image_size//8*128,hidden_size)
        # self.conv3=self.generate_conv_layer(10)
        # self.conv4=self.generate_conv_layer(1)
        # self.conv5=self.generate_conv_layer(1)

        
        
        size=sum(i//8*i//8*128 for i in [image_size])
        
        self.fc1 = nn.Sequential(
            nn.Linear(size, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(p=0.3)
            
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3)
            
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     # nn.LayerNorm(256),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.3)
        # )
        self.fc4 = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
    def generate_conv2d_layer(self,in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2),

            # 第二个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2),

            # 第三个卷积块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2)
        )
    def generate_conv3d_layer(self,in_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            #nn.AvgPool3d(kernel_size=2),

            # 第二个卷积块
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            #nn.AvgPool3d(kernel_size=2),

            # 第三个卷积块
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout3d(p=0.3),
            #nn.AvgPool3d(kernel_size=2)
        )
    def forward(self, images_seq1,images_seq2):
        
        #print(images_seq1.shape)
        images_seq1=self.conv1(images_seq1)
        images_seq1=images_seq1.flatten(start_dim=1)

        #print(images_seq2.shape)
        #images_seq2=images_seq2.permute(0, 2, 1, 3, 4)  
        #print(images_seq2.shape)
        B,T,C,H,W=images_seq2.shape
        images_seq2=images_seq2.view(B * T, C, H, W)
        #print(images_seq2.shape)
        images_seq2=self.conv2(images_seq2)
        images_seq2 = images_seq2.view(B, T, -1)  # [B, T, D] 128*16*16
        images_seq2=self.gru(images_seq2)[:,-1,:]

        #print(images_seq2.shape)
        
        
        
        
        x=self.fc1(images_seq1)
        x=torch.cat([x,images_seq2],dim=-1)
        #x=self.fc2(x)
        #x=self.fc3(x)
        x=self.fc4(x)
        return x


if __name__ == "__main__":
    net=NormalNet()
    import torch
    from torch.nn.utils.rnn import pad_sequence

    masks=[]
    datas=[]
    for i in range(10):
        Ns = [10, 5, 3, 1]
        batch = [torch.randint(0, 14, (n, 5)) for n in Ns]
        #print([batch[i].shape for i in range(len(batch))])
        padded = pad_sequence(batch, batch_first=True)  # [B, N_max, D]
        mask = (torch.arange(padded.shape[1])[None, :] < torch.tensor(Ns)[:, None]).detach()  # [B, N_max]
        masks.append(mask)
        datas.append(padded)
    print([datas[i].shape for i in range(len(datas))])
    datas=torch.stack(datas)

    print([masks[i].shape for i in range(len(masks))])
    masks=torch.stack(masks)
    print(datas.shape)
    print(masks.shape)
   
    
    state=torch.randn(10,2)

    images_seq1=torch.randn(10,1,128,128)
    


    
    output = net(state, datas, images_seq1, masks)
    # print(output)
    print(output.shape)