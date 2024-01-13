import torch
from torchvision import models

class Network():
    def __init__(self, device):

        self.model = models.resnet18(pretrained=True)
        i = 0
        for parameter in self.model.parameters():
            if i < 7:
                parameter.required_grad = False
        self.model.fc = torch.nn.Linear(in_features=512, out_features=6)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)