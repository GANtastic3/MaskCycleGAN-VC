# Code based on https://github.com/NVlabs/SPADE
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class TFAN_1D(nn.Module):
    """
    Implementation follows CycleGAN-VC3 paper: 
    Parameter choices for number of layers N=3, kernel_size in h is 5
    """

    def __init__(self, norm_nc, kernel_size=5, label_nc=80, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(norm_nc, affine=False)

        hidden_size = 128
        padding = kernel_size // 2

        mlp_layers = [nn.Conv1d(label_nc, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]
        for i in range(N - 1):
            mlp_layers += [nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]

        self.mlp_shared = nn.Sequential(*mlp_layers)
        self.mlp_gamma = nn.Conv1d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv1d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):
        # Step 1. Instance normalization of features
        normalized = self.param_free_norm(x)

        # print("Before TFAN interpolation")
        # print(segmap.shape, x.shape)

        # Step 2. Generate scale and bias conditioned on semantic map
        Bx, _, Qx, Tx = segmap.shape
        Bx, Qf, Tf = x.shape
        segmap = F.interpolate(segmap, size=(Qx, Tf), mode='nearest')
        segmap = segmap.squeeze(1)
        # print(segmap.shape)

        actv = self.mlp_shared(segmap)
        # print("actv: ", actv.shape)

        gamma = self.mlp_gamma(actv)
        # print("gamma: ", gamma.shape)
        beta = self.mlp_beta(actv)
        # print("beta: ", beta.shape)

        # Step 3. Apply scale and bias
        out = normalized * (1 + gamma) + beta
        # print(out.shape)
        return out


class TFAN_2D(nn.Module):
    """
    Implementation follows CycleGAN-VC3 paper: 
    Parameter choices for number of layers N=3, kernel_size=5
    """

    def __init__(self, norm_nc, kernel_size=5, label_nc=1, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        hidden_size = 128
        padding = kernel_size // 2
        
        mlp_layers = [nn.Conv2d(label_nc, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]
        for i in range(N - 1):
            mlp_layers += [nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding), nn.ReLU()]
        
        self.mlp_shared = nn.Sequential(*mlp_layers)
        self.mlp_gamma = nn.Conv2d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_size, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):
        # Step 1. Instance normalization of features
        normalized = self.param_free_norm(x)
        # print("normalized: ", normalized.shape)
        # print(segmap.shape)
        # print(x.shape)
 
        # Step 2. Generate scale and bias conditioned on semantic map
        Bx, _, Qx, Tx = segmap.shape
        Bx, Cf, Qf, Tf = x.shape
        segmap = F.interpolate(segmap, size=(Qf, Tf), mode='nearest')
        # segmap = F.interpolate(segmap, size=resize_shape[2:], mode='nearest')
        # print("shape after interpolate: ", segmap.shape)

        actv = self.mlp_shared(segmap)

        gamma = self.mlp_gamma(actv)
        # print("gamma: ", gamma.shape)
        beta = self.mlp_beta(actv)
        # print("beta: ", beta.shape)

        # Step 3. Apply scale and bias
        out = normalized * (1 + gamma) + beta
        # print(out.shape)
        return out