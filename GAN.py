import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 64 * 64),  
            nn.Tanh()  
        )

    def forward(self, noise):
        return self.net(noise).view(-1, 3, 64, 64) 

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 64 * 64, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self, img):
        return self.net(img.view(img.size(0), -1))
