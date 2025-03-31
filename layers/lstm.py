import torch
import torch.nn as nn





class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.W_x = nn.Linear(input_size, 3 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 2 * hidden_size)
        self.W_h2 = nn.Linear(hidden_size, 1 * hidden_size)
        self.hidden_size = hidden_size
        

    def forward(self, x):
        N, T, D = x.size()
        h = torch.zeros(N, self.hidden_size, device=x.device)

        outputs = []
        for t in range(T):
            xt = x[:, t, :] 
            rx,zx,hx = self.W_x(xt).chunk(3, dim=-1)
            rh,zh = self.W_h(h).chunk(2, dim=-1)
            r = torch.sigmoid(rx + rh)
            z = torch.sigmoid(zx + zh)
            hh = self.W_h2(h*r)
            h_ = torch.tanh(hx + hh)
            h = z*h_ + (1-z)*h
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        return outputs
            
        
if __name__ == "__main__":
    lstm = LSTM(128, 128)
    x = torch.randn(4, 10, 128)
    output = lstm(x)
    print(output.shape)




        