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
    def __init__(self,input_size_1,input_size_2,input_size_3):
        super().__init__()
        
        
        self.conv1 = nn.Sequential(
            # 第一个卷积块
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
        )# for images_seq1

        self.conv2 = nn.Sequential(
            # 第一个卷积块
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
        )# for images_seq2
        self.conv3=nn.Sequential(
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
        #self.resnet3 = ResNet([2, 2, 2, 2], channel_in=1)#(N, d_r)
        size=sum(i//8*i//8*128 for i in [input_size_1,input_size_2,input_size_3])
        self.fc1 = nn.Linear(size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

    def forward(self, images_seq1, images_seq2, images_seq3):
        

        # print(images_seq1.shape)
        # print(images_seq2.shape)
        # print(images_seq3.shape)
        #flat_imgs3 = images_seq3.view(T3 * N3, C3, H3, W3)
        #print(flat_imgs1.shape)
        
        flat_imgs1 = self.conv1(images_seq1)
        flat_imgs2 = self.conv2(images_seq2)
        flat_imgs3 = self.conv3(images_seq3)

        flat_imgs1 = flat_imgs1.flatten(start_dim=1)
        flat_imgs2 = flat_imgs2.flatten(start_dim=1)
        flat_imgs3 = flat_imgs3.flatten(start_dim=1)
        # print(flat_imgs1.shape)
        # print(flat_imgs2.shape)
        # print(flat_imgs3.shape)
        # print(torch.cat([flat_imgs1, flat_imgs2, flat_imgs3], dim=-1).shape)
        x = self.fc1(torch.cat([flat_imgs1, flat_imgs2, flat_imgs3], dim=-1))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        
        return x
    