from queue import Full
from tkinter import Scale
from turtle import forward
from numpy import pad, require
import torch
import torch.nn as nn


class ScaledConv2d(nn.Module):
    '''
    Here as stated in ProGAN, we do scaling for weights of 2D 
    convolution layers for equalized learning rate. First we set
    weigts of layers to be N(0, 1), and then do scaling by scaling 
    in_channels: input channels 
    out_channels: output channels (just like arguments of nn.Conv2d or other convolution layers)
    '''
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, factor = 2):
        super(ScaledConv2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.kernel_size = kernel_size 
        self.padding = padding 
        self.factor = self.factor 
        self.conv2d = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding = self.padding)
        self.conv2d.bias.fill_(0)
        self.conv2d.weight.data.normal_(0, 1)
        self.he_constant = 

        
    def forward(self, x):
        return self.conv2d(x*self.he_constant)



class ScaledLinear(nn.Module):
    '''
    As we did in convolution layers, here we also do weight scaling (more precisely input scaling 
    as it doesn't matter since they are just factors)
    '''
    def __init__(self, input_dim, output_dim, factor = 2):
        super(ScaledLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor 
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.linear_layer.bias.fill_(0)
        self.linear_layer.weight.data.normal_(0, 1)
        self.he_constant = 

    def forward(self, x):
        return self.linear_layer(x*self.he_constant)



class AdaIN(nn.Module):
    '''Adaptive instance normalization is for style transferring, it is two times faster 
    than patch based networks as mentioned Adaptive instance normalization paper. Here instead of using 
    mean, and standard deviation of w, we use affine transformations. '''
    def __init__(self, input_dim, output_dim, epsilon  = 10**(-5)):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.affine_transformations_scale= ScaledLinear(input_dim, output_dim)
        self.affine_transformations_bias = ScaledLinear(input_dim, output_dim)
    def forward(self, x, y):
        x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.std(x, dim = (2,3), keepdim=True) + self.epsilon
        x = (x - x_mean)/x_std
        y_scale = self.affine_transformations_scale(y).view(x.shape[0], x.shape[1], 1, 1)
        y_bias = self.affine_transformations_bias(y).view(x.shape[0], x.shape[1], 1, 1)

        x = x*y_scale + y_bias 
        return x 


class FullyConnected(nn.Module):
    '''
    Here we do our mapping of latent vector z to w, as explained in paper to 
    '''
    def __init__(self, latent_dim, num_layers, leaky_coefficient= 0.2):
        super(FullyConnected, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.leaky_coefficient = leaky_coefficient
        self.linear_layers = nn.Sequential(*([nn.Linear(latent_dim, latent_dim) for _ in range (num_layers)]))
        self.leaky_relu = nn.LeakyReLU(leaky_coefficient)
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.leaky_relu(self.linear_layers[i])
        return x 


class NoiseMultiplier(nn.Module):

    def __init__(self):
        super(NoiseMultiplier, self).__init__()
        self.module = nn.Conv2d(1, 1, 1, bias=False)
        self.module.weight.data.fill_(0)

    def forward(self, x):

        return self.module(x)



class ShynthesisNetwork(nn.Module):
    
    def __init__(self,
                 input_dim,
                 mapping_dim=512,
                 rgb_output_dim=3, 
                 num_mapping_layers=8, 
                 leaky = 0.2,
                 generationActivation=None,
                 phiTruncation=0.5,
                 gamma_avg=0.99):
        super().__init__()/avatars/463687786982277121/bb77ff39b1f2e5e832b90112bce5b069.webp
        '''As stated in the paper, dimension of latent vector is 512, and same for after applying
        fully connected mapping layer for the sake of simplicity. So input_dim and mapping_dim both are same'''
        self.input_dim = input_dim
        self.mapping_dim = mapping_dim
        self.mapping_forward_net = FullyConnected(input_dim, mapping_dim, num_mapping_layers)
        self.constant = nn.Parameter(torch.randn((1, mapping_dim, 4, 4) ), requires_grad=True) #Learnable constant
        self.rgb_output_dim= rgb_output_dim
        self.num_mapping_layers= num_mapping_layers
        self.leaky = leaky
        self.generationActivation= generationActivation
        self.phiTruncation= phiTruncation
        self.gamma_avg= gamma_avg
        self.RGBlayers = nn.Module
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        self.noiseModulators = nn.ModuleList()
        self.depthScales = [mapping_dim]
        self.activation = nn.LeakyReLU(0.2)
        self.Upsample = nn.UpsamplingBilinear2d()

        #We have different structure for convolutional layers in first conv block, that's why we create
        #two instance of AdaIN class 
        self.adain0 = AdaIN(mapping_dim, mapping_dim)
        self.adain1 = AdaIN(mapping_dim, mapping_dim)
        self.NoiseScaler0 = ScaledConv2d(1, 1, kernel_size=1, padding = 0)
        self.NoiseScaler1 = ScaledConv2d(1, 1, kernel_size=1, padding = 0)
        self.conv0 = ScaledConv2d(mapping_dim, mapping_dim)

    def set_alpha(self, alpha):
        '''By this function we change the value of merging factor'''
        
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha
    
    def add_scale(self, dimNewScale):
        lastDim = self.depthScales[-1]
        self.scaleLayers
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(ScaledConv2d(lastDim,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))

        self.scaleLayers[-1].append(AdaIN(self.mapping_dim, dimNewScale))
        self.scaleLayers[-1].append(ScaledConv2d(dimNewScale,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.mapping_dim, dimNewScale))
        self.toRGBLayers.append(ScaledConv2d(dimNewScale,
                                                self.rgb_output_dim,
                                                1,
                                                equalized=True,
                                                initBiasToZero=True))

        self.noiseModulators.append(nn.ModuleList())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.depthScales.append(dimNewScale)
    def forward(self, x):

        batchSize = x.size(0)
        mapping = self.mapping_forward_net((x - torch.mean(x))/(torch.std(x)+ 1e-8)) #normalization of layer to speed up learning. 
        if self.training:
            self.mean_w = self.gamma_avg * self.mean_w + (1-self.gamma_avg) * mapping.mean(dim=0, keepdim=True)

        if self.phiTruncation < 1:
            mapping = self.mean_w + self.phiTruncation * (mapping - self.mean_w)

        feature = self.constant.expand(batchSize, -1, 4, 4)
        feature = feature + self.NoiseScaler0(torch.randn((batchSize, 1, 4, 4), device=x.device))

        feature = self.activation(feature)
        feature = self.adain0(feature, mapping)
        feature = self.conv0(feature)
        feature = feature + self.NoiseScaler1(torch.randn((batchSize, 1, 4, 4), device=x.device))
        feature = self.activation(feature)
        feature = self.adain1(feature, mapping)

        for nLayer, group in enumerate(self.scaleLayers):

            noiseMod = self.noiseModulators[nLayer]
            feature = self.Upsample(feature)
            feature = group[0](feature) + noiseMod[0](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[1](feature, mapping)
            feature = group[2](feature) + noiseMod[1](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=x.device))
            feature = self.activation(feature)
            feature = group[3](feature, mapping)

            if self.alpha > 0 and nLayer == len(self.scaleLayers) -2:
                y = self.toRGBLayers[-2](feature)
                y = self.Upsample(y)

        feature = self.toRGBLayers[-1](feature)
        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            feature = self.alpha * y + (1.0-self.alpha) * feature

        if self.generationActivation is not None:
            feature = self.generationActivation(feature)

        return feature

    def getOutputSize(self):

        side =  2**(2 + len(self.toRGBLayers))
        return (side, side)

