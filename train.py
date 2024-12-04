import os
import torch
import numpy as np
from data import *
from ResNet import *
from VGG import *
from GAN import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

class Train:
    def __init__(self, root_path="CACD2000/", model_name="resnet50", number_classes=2000, 
                 path="model.pkl", loadPretrain=False):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        self.cacd_dataset = ImageData(root_path=root_path, label_path="data/label.npy", 
                                       name_path="data/name.npy", train_mode="train")
        
        self.model = resnet50(pretrained=loadPretrain, num_classes=number_classes, model_path=path)
        # # Initialize classification model
        # if model_name == "resnet50":
        #     self.model = resnet50(pretrained=loadPretrain, num_classes=number_classes, model_path=path)
        # elif model_name == "vgg16":
        #     self.model = vgg16(pretrained=loadPretrain, num_classes=number_classes, model_path=path)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"There are {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
            self.model.cuda()

        # Initialize GAN
        self.noise_dim = 100
        self.generator = Generator(self.noise_dim, img_channels=3).cuda()
        self.discriminator = Discriminator(img_channels=3).cuda()

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        self.gan_loss = nn.BCELoss()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        # Classification loss and optimizer
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epoch_num):
            print(f"Starting epoch {epoch + 1}/{self.epoch_num}")

            epoch_count = 0
            total_loss = 0
            dataloader = DataLoader(self.cacd_dataset, batch_size=self.batch_size, shuffle=True)

            for i_batch, sample_batch in enumerate(dataloader):
                print(f"Running: Epoch {epoch + 1}, Batch {i_batch + 1}")
 
                # Step 1: Load data and labels
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())

                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda(non_blocking=True))
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)

                # Step 2: Train Discriminator
                real_images = input_image  # Real images from dataset
                real_labels = torch.ones(self.batch_size, 1).cuda()
                fake_labels = torch.zeros(self.batch_size, 1).cuda()

                noise = torch.randn(self.batch_size, self.noise_dim).cuda()
                fake_images = self.generator(noise)  # Generate adversarial examples

                real_output = self.discriminator(real_images)
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
                adv_images = fake_images.detach()  # Freeze generator gradients
                output = self.model(adv_images)
                loss_adv = loss_function(output, target_label)

                optimizer.zero_grad()
                loss_adv.backward()
                optimizer.step()

                # Print results
                if i_batch % batch_display == 0:
                    pred_prob, pred_label = torch.max(output, dim=1)
                    batch_correct = (pred_label == target_label).sum().item() / self.batch_size
                    print(f"Epoch: {epoch + 1}, Batch: {i_batch + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, "
                          f"Classification Loss: {loss_adv.item()}, Accuracy: {batch_correct:.2f}")

            # Save model
            print(f"Saving model: Epoch {epoch + 1}, Average Loss: {total_loss / epoch_count:.4f}")
            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), f"{self.save_path}_{epoch + 1}.pth")
