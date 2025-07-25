import torch.nn as nn
import torch
device = 'cuda:0'
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1,1,kernel_size=(5,5))
    def forward(self, x):
        for i in range(50):
            self.conv1(x)#预热
        for i in range(50):
            x = self.conv1(x)
        return x

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.conv1 = nn.Conv2d(1,1,kernel_size=(1,5))
        self.conv2 = nn.Conv2d(1,1,kernel_size=(5,1))
    def forward(self, x):
        for i in range(50):
            self.conv1(x)#预热
        for i in range(50):
            x = self.conv1(x)
        return x
model = model().to(device)
model2 = model2().to(device)
if __name__ == "__main__":
    x = torch.randn(1,1,5120,5120).to(device)
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)

    start1.record()
    x1 = model(x).to(device)
    end1.record()
    torch.cuda.synchronize()
    print(start1.elapsed_time(end1) / 50)

    start2.record()
    x2 = model2(x).to(device)
    end2.record()
    torch.cuda.synchronize()
    print(start2.elapsed_time(end2) / 50)