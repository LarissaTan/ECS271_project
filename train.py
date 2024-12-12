import os
import torch
import numpy as np
from data import *
from ResNet import *
from GAN import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class Train:
    def __init__(self, root_path="CACD2000/", model_name="resnet50", number_classes=2000, 
                 path="model.pkl", loadPretrain=False):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.save_path = path
        self.cacd_dataset = ImageData(root_path=root_path, label_path="data/label.npy", 
                                      name_path="data/name.npy", train_mode="train")
        
        self.model = resnet50(pretrained=loadPretrain, num_classes=number_classes, model_path=path).to(self.device)

        self.noise_dim = 100
        self.generator = Generator(self.noise_dim).to(self.device)
        self.discriminator = Discriminator(img_channels=3).to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.gan_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, save_freq=1):
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate

        for ep in range(self.epoch_num):
            print(f"Starting epoch {ep + 1}/{self.epoch_num}")

            dataloader = DataLoader(self.cacd_dataset, batch_size=self.batch_size, shuffle=True)

            for i_batch, sample_batch in enumerate(dataloader):

                real_images = sample_batch['image'].to(self.device)
                labels_batch = sample_batch['label'].to(self.device)
                
                
                labels_batch = labels_batch.long().view(-1)
                
                batch_size = real_images.size(0)

                noise = torch.randn(batch_size, self.noise_dim, device=self.device)

                # =================== 训练判别器 ===================
                fake_images = self.generator(real_images, noise)
                real_output = self.discriminator(real_images)
                fake_output = self.discriminator(fake_images.detach())

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                d_loss_real = self.gan_loss(real_output, real_labels)
                d_loss_fake = self.gan_loss(fake_output, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # =================== 训练生成器 ===================
                fake_images = self.generator(real_images, noise)
                fake_output = self.discriminator(fake_images)

                g_loss = self.gan_loss(fake_output, real_labels)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # =================== 训练分类模型 ===================
                adv_images = fake_images.detach()
                classification_output = self.model(adv_images)
                cls_loss = self.classification_loss(classification_output, labels_batch)

                self.model_optimizer.zero_grad()
                cls_loss.backward()
                self.model_optimizer.step()

                # =================== 每100个batch打印日志并输出生成的图片 ===================
                if i_batch % 100 == 0:
                    pred_prob, pred_label = torch.max(classification_output, dim=1)
                    batch_correct = (pred_label == labels_batch).sum().item() / batch_size
                    print(f"Epoch: {ep + 1}, Batch: {i_batch + 1}/{len(dataloader)}, "
                          f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                          f"Cls Loss: {cls_loss.item():.4f}, Accuracy: {batch_correct:.4f}")
                    
                    
                    with torch.no_grad():
                        sample_noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                        gen_samples = self.generator(real_images, sample_noise)
                        vutils.save_image(gen_samples, f"fake_samples_epoch_{ep+1}_batch_{i_batch+1}.png", normalize=True, nrow=8)


            if ep % save_freq == 0:
                torch.save(self.model.state_dict(), f"{self.save_path}_{ep + 1}.pth")
                print(f"Model saved at epoch {ep + 1}.")
