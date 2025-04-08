import torch
import torch.nn as nn
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.setTransfer import SetTransformer
from layers.lstm import GRU


class NormalNet(nn.Module):
    def __init__(self,hidden_size=128,image_size=128):
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

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第二个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第三个卷积块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第二个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第三个卷积块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第二个卷积块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第三个卷积块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        size=sum(i//8*i//8*128 for i in [image_size,image_size,image_size])
        self.fc1 = nn.Sequential(
            nn.Linear(size, 1024),
            nn.LayerNorm(1024),
            nn.Tanh()
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Tanh()
            
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
    def forward(self, images_seq1,images_seq2,images_seq3):
        

        images_seq1=self.conv1(images_seq1)
        images_seq1=images_seq1.flatten(start_dim=1)
        images_seq2=self.conv2(images_seq2)
        images_seq2=images_seq2.flatten(start_dim=1)
        images_seq3=self.conv3(images_seq3)
        images_seq3=images_seq3.flatten(start_dim=1)
        
        x=torch.cat([images_seq1,images_seq2,images_seq3],dim=-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
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