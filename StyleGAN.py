from __future__ import print_function
from base64 import b16decode
from cmath import e
from distutils.command.config import config
from functools import partial
import argparse
# from distutils.command.config import config
import os 
import random
from turtle import circle
#from termios import B4000000
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn 
import torch.functional as F
import torch.optim as optim 
import torch.utils.data 
import torchvision.datasets as dset 
import torchvision.transforms as transforms 
import torchvision.utils as vutils 
import numpy as np 


#device that we will use train, and computation of tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#initializing weights of models. We will use the DCGAN weight initialization.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#Necessary constants 
lr = 0.001
beta1 = 0.9 
epochs = 5
image_size = 256
batch_size = 64
workers = 0
#Dataset:
dataroot = "/Users/coll1ns/Documents/StyleGAN/anime folder"
datas = dset.ImageFolder(root = dataroot,  
                    transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
                    )
# Create the dataloader
dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #First we make MLP for transforming latent vector space. 
        self.main = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    #1 to 2
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    #2 to 3
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    #3 to 4
                     nn.Linear(512, 512),
                    nn.ReLU(),
                    #4 to 5
                     nn.Linear(512, 512),
                    nn.ReLU(),
                    #5 to 6
                     nn.Linear(512, 512),
                    nn.ReLU(),
                    #6 to 7
                     nn.Linear(512, 512),
                    nn.ReLU(),
                    #7 to 8
                     nn.Linear(512, 512)
                    )
        #Learnable constant
        self.constant = nn.Parameter(torch.randn(1, 512 , 4, 4, device = device))
        #Convolution networks for convolution blocks, we have 10 blocks
        self.conv = nn.Sequential(*([nn.Conv2d(512, 512, kernel_size = 3, padding=1) for i in range (7)]))
        # noise factors: In paper it is written that noise factors are also learnable parameters for each channel
        self.noise_factors  = nn.Parameter(torch.randn((1, 512, 1, 1, 13), device = device)) #fix this line!
        #affine transformation parameters. We can basically pass transformed vector through linear layers 
        #without activation function.
        self.affine_transformation_mean = nn.Sequential(*([nn.Linear(512, 512) for _ in range(13)]))
        self.affine_transformation_std = nn.Sequential(*([nn.Linear(512, 512) for _ in range(13)]))
        #I used elu activation function which is better if we consider that LeakyReLu to be more linear activation function. 
        self.elu = nn.ELU()
        #upsampling, as in mentioned paper, we use bilinear upsampling which is supposed to give better results
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        #noise, we produce noise in shape of current state of feature maps. 
        self.noise = torch.randn_like
        self.flatten = nn.Flatten()
        self.conv_last = nn.Conv2d(512, 3, kernel_size=1, stride=1 )

    def forward(self, z):
        def AdaIN(self, x, i):
            # print(z.size())
            # z = self.flatten(z)
            w = self.main(z)
            epsilon = 10**(-6) #for numerical stability.
            mean_x = torch.mean(x, dim = (2, 3), keepdim= True)
            std_x = torch.std(x, dim = (2, 3), keepdim= True) + epsilon
            mean_y = self.affine_transformation_mean[i](w).view(x.shape[0], x.shape[1], 1, 1)
            std_y = self.affine_transformation_std[i](w).view(x.shape[0], x.shape[1], 1, 1)
            # mean_y = torch.full((x.shape[0], x.shape[1], 1, 1), torch.mean(y).item())
            # std_y = torch.full((x.shape[0], x.shape[1], 1, 1), torch.std(y).item())
            x = (x-mean_x)/std_x
            return std_y*x+mean_y
        x = self.constant
        for i in range(6):
            # if i != 0:
            x = self.upsample(x)
            x = x + self.noise(x)*self.noise_factors[:, :, :, :, 2*i]
            x = AdaIN(self, x, 2*i)
            # print(x.size())
            x = self.conv[i](x)
            # print(x.size())
            x = self.elu(x)
            x = x+self.noise(x)*self.noise_factors[:, :, :, :,2*i+1]
            x = AdaIN(self, x, 2*i+1)
        print(x.size(), "after for loop")
        x  = self.conv_last(x)
        print(x.size(), "after convolution")
        return x

gen = Generator().to(device)
# gen.apply(weights_init) #applying weight initialization to model parameters.
print(gen)

gen.apply(weights_init)
# class Minibatch(nn.Module):
#     pass

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 1)
        self.convs= nn.Sequential(
                                    nn.Conv2d(16, 16, 3, padding= 1 ),
                                    nn.Conv2d(16, 32, 3, padding= 1),
                                    nn.Conv2d(32, 32, 3, padding = 1),
                                    nn.Conv2d(32, 64, 3, padding= 1),
                                    nn.Conv2d(64, 64, 3, padding= 1),
                                    nn.Conv2d(64, 128, 3, padding= 1),
                                    nn.Conv2d(128, 128, 3, padding= 1),
                                    nn.Conv2d(128, 256, 3, padding= 1),
                                    nn.Conv2d(256, 256, 3, padding= 1),
                                    nn.Conv2d(256, 512, 3, padding= 1),
                                    nn.Conv2d(512, 512, 3, padding= 1),
                                    nn.Conv2d(512, 512, 3, padding= 1)
                                    )
        self.downsample = nn.Upsample(scale_factor=0.5, mode = 'bilinear')
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv3 = nn.Conv2d(512, 512, 4)
        self.linear = nn.Linear(512, 1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        #self.minibatch = nn.Minibatch()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        print(x.size())
        n = 16
        for i in range(6): 
            x = self.leaky(self.convs[2*i](x))
            print(x.size(), "current error")
            x = self.leaky(self.convs[2*i+1](x))
            x = self.downsample(x)
        #x = self.minibatch(x)
        x = self.leaky(self.conv2(x))
        print(x.size())
        x = self.leaky(self.conv3(x))
        flattened_x = self.flatten(x)
        print(flattened_x.size())
        return self.softmax(self.linear(flattened_x))





dis = Discriminator().to(device)

print(dis)

dis.apply(weights_init)

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

optimizerG = optim.Adam(gen.parameters(), lr = lr, betas = (beta1, 0.99))
optimizerD = optim.Adam(dis.parameters(), lr = lr, betas = (beta1, 0.99))

img_list = []
G_losses = 0
D_losses = 0
iters = 0 

fixed_noise = z = torch.randn(1, 512, 1, 1,  device=device)

print("Starting Training Loop")



for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        dis.zero_grad()

        #all real

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,dtype=torch.float, device = device)
        output = dis(real_cpu).view(-1)
        print(output)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        #all fake 
        z = torch.randn(1, 512, device=device)
        fake = gen(z)
        label.fill_(fake_label)
        print(label.size(), "label size")
        output = dis(fake.detach()).view(-1)
        print(output.size(), "output size")
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_fake + errD_real
        optimizerD.step()

        gen.zero_grad()
        label.fill_(real_label)
        output = dis(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            if iters < 10000:
                print('#####')
                print(iters)
                print('#####')
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1







