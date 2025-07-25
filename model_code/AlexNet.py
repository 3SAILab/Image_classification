import torch
import torch.nn as nn
import torch.functional as F
import time
class AlexNet(nn.Module):
    def __init__(self, num_classes, visual_features=False) -> None:
        super().__init__()
        self.visual_features = visual_features
        self.features = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(128,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
        )
        self.classfier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*2*2,512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        if self.visual_features:
            feature_list = []
            for name, module in self.features.named_children():
                x = module(x)
                if name in ['0', '3', '6']:
                    feature_list.append(x)
            return feature_list
        else:
            x = self.features(x)
            x = torch.flatten(x, start_dim = 1)
            x = self.classfier(x)
            return x

if __name__ == "__main__":
    input_tensor = torch.rand(128, 3, 224, 224).to("cuda")
    model = AlexNet(25).to("cuda")
    for i in range(50):
        result = model(input_tensor).to("cuda")
    start_time = time.time() 
    for i in range(100):
        result = model(input_tensor).to("cuda")
    end_time = time.time() - start_time
    print(end_time / 100)