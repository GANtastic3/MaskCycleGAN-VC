import torch.nn as nn
import torch
from mask_cyclegan_vc.tfan_module import TFAN_1D, TFAN_2D


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


##########################################################################################
class Generator(nn.Module):
    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Generator, self).__init__()
        Cx, Tx = input_shape
        self.flattened_channels = (Cx // 4) * residual_in_channels
        # 2D Conv Layer
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=residual_in_channels // 2,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=2,
                                     out_channels=residual_in_channels // 2,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsample Layer
        self.downSample1 = downSample_Generator(in_channels=residual_in_channels // 2,
                                                out_channels=residual_in_channels,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        self.downSample2 = downSample_Generator(in_channels=residual_in_channels,
                                                out_channels=residual_in_channels,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Conv1d(in_channels=self.flattened_channels,
                                         out_channels=residual_in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv2dto1dLayer_tfan =  nn.InstanceNorm1d(num_features=residual_in_channels,affine=True)
        # self.conv2dto1dLayer_tfan = TFAN_1D(residual_in_channels)

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Conv1d(in_channels=residual_in_channels,
                                         out_channels=self.flattened_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv1dto2dLayer_tfan = nn.InstanceNorm1d(num_features=self.flattened_channels, affine=True)
        # self.conv1dto2dLayer_tfan = TFAN_1D(self.flattened_channels)

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 4,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.upSample1_tfan = TFAN_2D(residual_in_channels)
        self.glu = GLU()

        self.upSample2 = self.upSample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 2,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        self.upSample2_tfan = TFAN_2D(residual_in_channels // 2)

        self.lastConvLayer = nn.Conv2d(in_channels=residual_in_channels // 2,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       GLU())

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(	
                                           num_features=out_channels // 4,	
                                           affine=True),	
                                       GLU())
        return self.convLayer

    def forward(self, x, mask):
        # GLU
        x = torch.stack((x*mask, mask),dim=1)
        # print("Generator forward input: ", x.shape)
        # x = x.unsqueeze(1)
        # print("Generator forward input: ", x.shape)

        conv1 = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))
        # print("Generator forward conv1: ", conv1.shape)

        # DownloadSample
        downsample1 = self.downSample1(conv1)
        # print("Generator forward downsample1: ", downsample1.shape)
        downsample2 = self.downSample2(downsample1)
        # print("Generator forward downsample2: ", downsample2.shape)

        # 2D -> 1D
        # reshape
        reshape2dto1d = downsample2.view(downsample2.size(0), self.flattened_channels , 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        # print("Generator forward reshape2dto1d: ", reshape2dto1d.shape)
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        # print("Generator forward conv2dto1d_layer: ", conv2dto1d_layer.shape)
        conv2dto1d_layer = self.conv2dto1dLayer_tfan(conv2dto1d_layer)
        # print("Generator forward conv2dto1d_layer_tfan: ", conv2dto1d_layer.shape)

        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # print("Generator forward residual_layer_6: ", residual_layer_6.shape)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        # print("Generator forward conv1dto2d_layer: ", conv1dto2d_layer.shape)
        conv1dto2d_layer = self.conv1dto2dLayer_tfan(conv1dto2d_layer)

        # Reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)
        # print("Generator forward reshape1dto2d: ", reshape1dto2d.shape)

        # UpSample
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        # print("Generator forward upsample_layer_1: ", upsample_layer_1.shape)
        # upsample_layer_1 = self.upSample1_tfan(upsample_layer_1)
        # print("Generator forward upsample_layer_1: ", upsample_layer_1.shape)
        # upsample_layer_1 = self.glu(upsample_layer_1)

        upsample_layer_2 = self.upSample2(upsample_layer_1)
        # print("Generator forward upsample_layer_2: ", upsample_layer_2.shape)
        # upsample_layer_2 = self.upSample2_tfan(upsample_layer_2, x)
        # upsample_layer_2 = self.glu(upsample_layer_2)

        output = self.lastConvLayer(upsample_layer_2)
        # print("Generator forward output: ", output.shape)
        output = output.squeeze(1)
        # print("Generator forward output: ", output.shape)
        return output


##########################################################################################
# PatchGAN
class Discriminator(nn.Module):
    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=residual_in_channels // 2,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        GLU())

        # DownSample Layer
        self.downSample1 = self.downSample(in_channels=residual_in_channels // 2,
                                           out_channels=residual_in_channels,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=residual_in_channels,
                                           out_channels=residual_in_channels * 2,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downSample(in_channels=residual_in_channels * 2,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)

        self.downSample4 = self.downSample(in_channels=residual_in_channels * 4,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[1, 10],
                                           stride=(1, 1),
                                           padding=(0, 2))

        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=residual_in_channels * 4,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  GLU())
        return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        # print("Discriminator forward input: ", input.shape)
        conv_layer_1 = self.convLayer1(input)
        # print("Discriminator forward conv_layer_1: ", conv_layer_1.shape)
        downsample1 = self.downSample1(conv_layer_1)
        # print("Discriminator forward downsample1: ", downsample1.shape)
        downsample2 = self.downSample2(downsample1)
        # print("Discriminator forward downsample2: ", downsample2.shape)
        downsample3 = self.downSample3(downsample2)
        # print("Discriminator forward downsample3: ", downsample3.shape)
        # print("Discriminator forward downsample3: ", downsample3.shape)
        output = torch.sigmoid(self.outputConvLayer(downsample3))
        # print("Discriminator forward output: ", output.shape)
        return output


if __name__ == '__main__':
    import sys
    import numpy as np

    # args = sys.argv
    # print(args)
    # if len(args) > 1:
    #     if args[1] == "g":
    #         generator = Generator()
    #         print(generator)
    #     elif args[1] == "d":
    #         discriminator = Discriminator()
    #         print(discriminator)

    #     sys.exit(0)

    # Generator Dimensionality Testing
    np.random.seed(0)

    residual_in_channels = 256
    # input = np.random.randn(2, 80, 64)
    input = np.random.randn(2, 80, 64)
    input = torch.from_numpy(input).float()
    print("Generator input: ", input.shape)
    mask = torch.ones_like(input)
    generator = Generator(input.shape[1:], residual_in_channels)
    output = generator(input, mask)
    print("Generator output shape: ", output.shape)

    # Discriminator Dimensionality Testing
    discriminator = Discriminator(input.shape[1:], residual_in_channels)
    output = discriminator(output)
    print("Discriminator output shape ", output.shape)