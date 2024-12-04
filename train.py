import os
import torch
import numpy as np
from data import *
from ResNet import *
from VGG import *
from GAN import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

class Train:
    def __init__(self, root_path="CACD2000/", model_name="resnet50", number_classes=2000, 
                 path="model.pkl", loadPretrain=False):
        """
        Initialize Dataset, Model, and others
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.save_path = path
        self.cacd_dataset = ImageData(root_path=root_path, label_path="data/label.npy", 
                                       name_path="data/name.npy", train_mode="train")
        
        # Initialize classification model
        self.model = resnet50(pretrained=loadPretrain, num_classes=number_classes, model_path=path).to(self.device)

        # Initialize GAN components
        self.noise_dim = 100
        self.generator = Generator(self.noise_dim, img_channels=3).to(self.device)
        self.discriminator = Discriminator(img_channels=3).to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        self.gan_loss = nn.BCELoss()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Train the models (GAN and classification)
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate

        # Classification loss and optimizer
        loss_function = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch_num):
            print(f"Starting epoch {epoch + 1}/{self.epoch_num}")

            dataloader = DataLoader(self.cacd_dataset, batch_size=self.batch_size, shuffle=True)

            for i_batch, sample_batch in enumerate(dataloader):
                print(f"Running: Epoch {epoch + 1}, Batch {i_batch + 1}")
            
                # Step 1: Load data and labels
                images_batch = sample_batch['image'].to(self.device)
                labels_batch = sample_batch['label'].to(self.device).long().squeeze()

                print(f"Image batch shape: {images_batch.shape}, Labels shape: {labels_batch.shape}")

                # Step 2: Train Discriminator
                real_labels = torch.ones(images_batch.size(0), 1).to(self.device)
                fake_labels = torch.zeros(images_batch.size(0), 1).to(self.device)

                noise = torch.randn(images_batch.size(0), self.noise_dim).to(self.device)
                fake_images = self.generator(noise)

                real_output = self.discriminator(images_batch)
                fake_output = self.discriminator(fake_images.detach())

                d_loss_real = self.gan_loss(real_output, real_labels)
                d_loss_fake = self.gan_loss(fake_output, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Step 3: Train Generator
                fake_output = self.discriminator(fake_images)
                g_loss = self.gan_loss(fake_output, real_labels)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Step 4: Train Classification Model with Adversarial Examples
                adv_images = fake_images.detach()
                output = self.model(adv_images)

                print(f"Output shape: {output.shape}, Labels shape: {labels_batch.shape}")
                loss_adv = loss_function(output, labels_batch)

                optimizer.zero_grad()
                loss_adv.backward()
                optimizer.step()

                # Print results
                if i_batch % batch_display == 0:
                    pred_prob, pred_label = torch.max(output, dim=1)
                    batch_correct = (pred_label == labels_batch).sum().item() / labels_batch.size(0)
                    print(f"Epoch: {epoch + 1}, Batch: {i_batch + 1}, D Loss: {d_loss.item():.4f}, "
                        f"G Loss: {g_loss.item():.4f}, Classification Loss: {loss_adv.item():.4f}, "
                        f"Accuracy: {batch_correct:.2f}")


            # Save model
            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), f"{self.save_path}_{epoch + 1}.pth")
                print(f"Model saved at epoch {epoch + 1}.")
