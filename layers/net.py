import torch.nn as nn
import torch
import os
if __name__ == "__main__":
    from resnet import ResNet
    from transformer import GameTransformer
else:
    from layers.resnet import ResNet
    from layers.transformer import GameTransformer

class Net(nn.Module):
    def __init__(self,d_t,d_r,d_state, hidden_dim, num_layers,action_dim,vocab_size,embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(80, embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)
        self.resnet1 = ResNet([1, 1, 1, 1], channel_in=1)#(N, d_r)
        self.resnet2 = ResNet([1, 1, 1, 1], channel_in=1)#(N, d_r)
        #self.resnet3 = ResNet([2, 2, 2, 2], channel_in=1)#(N, d_r)
        
        self.state_proj=nn.Linear(d_state+embedding_dim*vocab_size, d_t)
        self.transformer = GameTransformer(d_t)
        self.lstm = nn.LSTM(input_size=d_t + d_r*2, hidden_size=hidden_dim, num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, images_seq1, images_seq2, state_seq,obj_ids):
        N,T,  C, H, W = images_seq1.shape
        N2,T2, C2, H2, W2 = images_seq2.shape
        #T3, N3, C3, H3, W3 = images_seq3.shape

        flat_imgs1 = images_seq1.view(T * N, C, H, W)
        flat_imgs2 = images_seq2.view(T2 * N2, C2, H2, W2)
        #flat_imgs3 = images_seq3.view(T3 * N3, C3, H3, W3)
        #print(flat_imgs1.shape)
        
        flat_imgs1 = self.resnet1(flat_imgs1)
        flat_imgs2 = self.resnet2(flat_imgs2)
        #flat_imgs3 = self.resnet3(flat_imgs3)
        flat_imgs1 = flat_imgs1.view(N,T,  -1)#(N, T, d_r)
        flat_imgs2 = flat_imgs2.view(N2,T2, -1)#(N2, T2, d_r)
        #flat_imgs3 = flat_imgs3.view(T3, N3, -1)#(T, N, d_r)
        #print(flat_imgs1.shape)

        #print(obj_ids.shape)
        obj_ids_embedding = self.embedding(obj_ids)
        # print(obj_ids_embedding.shape)
        # print(obj_ids.shape)
        # print(state_seq.shape)
        N,T,d_id,d_emb=obj_ids_embedding.shape
        #print(obj_ids_embedding.shape)
        embedding_flatten=obj_ids_embedding.view(N,T,d_id*d_emb)
        #print(torch.cat([state_seq, embedding_flatten], dim=-1).shape)
        state_proj = self.state_proj(torch.cat([state_seq, embedding_flatten], dim=-1))
        #print(state_proj.shape)
        transformer_out = self.transformer(state_proj) #(N, T, d_t)
        #print(transformer_out.shape)



        combined = torch.cat([transformer_out, flat_imgs1, flat_imgs2], dim=-1)  


        lstm_out, _ = self.lstm(combined)
        final_out = lstm_out[:, -1, :] 

        return final_out
    
if __name__ == "__main__":
    device = torch.device("cpu")
    net = Net(d_t=128, d_r=512, d_state=20, hidden_dim=128, num_layers=2, action_dim=10)
    net.to(device)
    for i in range(1):
        images_seq1 = torch.randn(1, 5, 1, 224, 224).to(device)
        images_seq2 = torch.randn(1, 5, 1, 224, 224).to(device)
        #images_seq3 = torch.randn(5, 2, 1, 224, 224).to(device)
        state_seq = torch.randn(1,5, 20).to(device)
        print(net(images_seq1, images_seq2, state_seq))
            