import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # 输出 [batch_size, 1, H, W]
            nn.AdaptiveAvgPool2d((1, 1)),  # 将 [H, W] 压缩到 [1, 1]
        )
        self.sigmoid = nn.Sigmoid()  # 输出概率值

    def forward(self, x):
        print(f"Discriminator Input Shape: {x.shape}")
        x = self.main(x)  # 输出 [batch_size, 1, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 1]
        print(f"Discriminator Output Shape: {x.shape}")
        x = self.sigmoid(x)
        return x


import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # 输出 [batch_size, 1, H, W]
            nn.AdaptiveAvgPool2d((1, 1)),  # 将 [H, W] 压缩到 [1, 1]
        )
        self.sigmoid = nn.Sigmoid()  # 输出概率值

    def forward(self, x):
        print(f"Discriminator Input Shape: {x.shape}")
        x = self.main(x)  # 输出 [batch_size, 1, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 1]
        print(f"Discriminator Output Shape: {x.shape}")
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Linear(noise_dim, 224 * 224)  # 将噪声映射到 224x224 的大小
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, img, noise):
        batch_size, channels, height, width = img.size()
        print(f"Generator Input Shapes -> Image: {img.shape}, Noise: {noise.shape}")
        if noise.size(1) != self.noise_dim:
            raise ValueError(f"Noise shape mismatch: Expected {self.noise_dim}, got {noise.size(1)}")

        # 将噪声映射为与224x224匹配的形状
        noise = self.fc(noise).view(batch_size, 1, height, width)
        print(f"Noise reshaped to: {noise.shape}")

        # 将噪声作为额外通道拼接到图像上
        x = torch.cat([img, noise], dim=1)
        print(f"Concatenated Input Shape: {x.shape}")
        x = self.conv_blocks(x)
        print(f"Generator Output Shape: {x.shape}")
        return x

