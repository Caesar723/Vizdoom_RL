import torch
import torch.nn as nn
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.setTransfer import SetTransformer
from layers.lstm import LSTM


class LabelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.set_transformer = SetTransformer(embed_dim*2, embed_dim)

    def forward(self, label_tensor,mask):  # [B, N, 5]
        class_id = label_tensor[..., 0].long()           # [B, N]
        bbox = label_tensor[..., 1:].float()                  # [B, N, 4]
        # print(class_id.shape)
        # print(bbox.shape)
        class_vec = self.class_embed(class_id)           # [B, N, D]
        bbox_vec = self.bbox_mlp(bbox)                   # [B, N, D]

        label_vec = torch.cat((class_vec, bbox_vec), dim=-1)  # Concatenate along the last dimension
        # print(label_vec.shape)
        # print(mask.shape)
        return self.set_transformer(label_vec,mask)           # [B, D]

class NormalNet(nn.Module):
    def __init__(self,hidden_size=128,image_size=128):
        super().__init__()
        self.hidden_size=hidden_size

        self.lstm=LSTM(hidden_size,hidden_size)

        self.state_encoder=nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        self.label_encoder=LabelEncoder(14, hidden_size)
        self.conv=nn.Sequential(
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
        self.image_linear=nn.Linear(image_size//8*image_size//8*128,1024)
        self.fc1 = nn.Linear(hidden_size*2, 512)
        self.fc2 = nn.Linear(1024+512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)


    def forward(self, state, label_tensor,images_seq1,mask=None):
        print(label_tensor.shape)
        B,T,N,L=label_tensor.size()
        
        label_tensor=label_tensor.view(T*B,N,L)
        print(label_tensor.shape)
        mask=mask.view(T*B,N)
        label_tensor=self.label_encoder(label_tensor,mask)
        label_tensor=label_tensor.view(B,T,self.hidden_size)
        print(label_tensor.shape)
        label_tensor=self.lstm(label_tensor)[:,-1,:]
        print(label_tensor.shape)
        print(state.shape)
        state=self.state_encoder(state)
        print(state.shape)

        
        images_seq1=self.conv(images_seq1)
        images_seq1=images_seq1.flatten(start_dim=1)
        images_seq1=self.image_linear(images_seq1)

        print(state.shape)
        print(label_tensor.shape)
        print(images_seq1.shape)
        x=torch.cat([state,label_tensor],dim=-1)
        x=self.fc1(x)
        x=self.fc2(torch.cat([x,images_seq1],dim=-1))
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