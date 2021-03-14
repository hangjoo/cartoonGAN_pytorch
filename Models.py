import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import init_weights


# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, ... )


class Generator(nn.Module):
    def __init__(self, init_weights_path=None):
        super(Generator, self).__init__()

        # Activation Functions
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        ## First Block
        self.pad_01_1 = nn.ReflectionPad2d(3)
        self.conv_01_1 = nn.Conv2d(3, 64, 7)
        self.in_norm_01_1 = InstanceNormalization(64)
        # ReLU

        ## Down Convolution Block 1
        self.conv_02_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv_02_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in_norm_02_1 = InstanceNormalization(128)
        # ReLU

        ## Down Convolution Block 2
        self.conv_03_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv_03_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in_norm_03_1 = InstanceNormalization(256)
        # ReLU

        ## Residual Block 1
        self.pad_04_1 = nn.ReflectionPad2d(1)
        self.conv_04_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_04_1 = InstanceNormalization(256)
        # ReLU
        self.pad_04_2 = nn.ReflectionPad2d(1)
        self.conv_04_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_04_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 2
        self.pad_05_1 = nn.ReflectionPad2d(1)
        self.conv_05_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_05_1 = InstanceNormalization(256)
        # ReLU
        self.pad_05_2 = nn.ReflectionPad2d(1)
        self.conv_05_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_05_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 3
        self.pad_06_1 = nn.ReflectionPad2d(1)
        self.conv_06_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_06_1 = InstanceNormalization(256)
        # ReLU
        self.pad_06_2 = nn.ReflectionPad2d(1)
        self.conv_06_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_06_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 4
        self.pad_07_1 = nn.ReflectionPad2d(1)
        self.conv_07_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_07_1 = InstanceNormalization(256)
        # ReLU
        self.pad_07_2 = nn.ReflectionPad2d(1)
        self.conv_07_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_07_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 5
        self.pad_08_1 = nn.ReflectionPad2d(1)
        self.conv_08_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_08_1 = InstanceNormalization(256)
        # ReLU
        self.pad_08_2 = nn.ReflectionPad2d(1)
        self.conv_08_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_08_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 6
        self.pad_09_1 = nn.ReflectionPad2d(1)
        self.conv_09_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_09_1 = InstanceNormalization(256)
        # ReLU
        self.pad_09_2 = nn.ReflectionPad2d(1)
        self.conv_09_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_09_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 7
        self.pad_10_1 = nn.ReflectionPad2d(1)
        self.conv_10_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_10_1 = InstanceNormalization(256)
        # ReLU
        self.pad_10_2 = nn.ReflectionPad2d(1)
        self.conv_10_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_10_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Residual Block 8
        self.pad_11_1 = nn.ReflectionPad2d(1)
        self.conv_11_1 = nn.Conv2d(256, 256, 3)
        self.in_norm_11_1 = InstanceNormalization(256)
        # ReLU
        self.pad_11_2 = nn.ReflectionPad2d(1)
        self.conv_11_2 = nn.Conv2d(256, 256, 3)
        self.in_norm_11_2 = InstanceNormalization(256)
        # Elementwise Sum

        ## Up Convolution Block 1
        self.deconv_12_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv_12_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in_norm_12_1 = InstanceNormalization(128)
        # ReLU

        ## Up Convolution Block 2
        self.deconv_13_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv_13_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.in_norm_13_1 = InstanceNormalization(64)
        # ReLU

        ## Final Block
        self.pad_14_1 = nn.ReflectionPad2d(3)
        self.conv_14_1 = nn.Conv2d(64, 3, 7)
        # tanh

        # Initialize weights
        if init_weights_path is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(init_weights_path)["model_state_dict"])

    def forward(self, x):
        # First Block
        f1 = self.ReLU(self.in_norm_01_1(self.conv_01_1(self.pad_01_1(x))))

        # Down Convolution Blocks
        d1 = self.ReLU(self.in_norm_02_1(self.conv_02_2(self.conv_02_1(f1))))
        d2 = self.ReLU(self.in_norm_03_1(self.conv_03_2(self.conv_03_1(d1))))

        # Residual Blocks
        r1_1 = self.ReLU(self.in_norm_04_1(self.conv_04_1(self.pad_04_1(d2))))
        r1_2 = self.in_norm_04_2(self.conv_04_2(self.pad_04_2(r1_1))) + d2

        r2_1 = self.ReLU(self.in_norm_05_1(self.conv_05_1(self.pad_05_1(r1_2))))
        r2_2 = self.in_norm_05_2(self.conv_05_2(self.pad_05_2(r2_1))) + r1_2

        r3_1 = self.ReLU(self.in_norm_06_1(self.conv_06_1(self.pad_06_1(r2_2))))
        r3_2 = self.in_norm_06_2(self.conv_06_2(self.pad_06_2(r3_1))) + r2_2

        r4_1 = self.ReLU(self.in_norm_07_1(self.conv_07_1(self.pad_07_1(r3_2))))
        r4_2 = self.in_norm_07_2(self.conv_07_2(self.pad_07_2(r4_1))) + r3_2

        r5_1 = self.ReLU(self.in_norm_08_1(self.conv_08_1(self.pad_08_1(r4_2))))
        r5_2 = self.in_norm_08_2(self.conv_08_2(self.pad_08_2(r5_1))) + r4_2

        r6_1 = self.ReLU(self.in_norm_09_1(self.conv_09_1(self.pad_09_1(r5_2))))
        r6_2 = self.in_norm_09_2(self.conv_09_2(self.pad_09_2(r6_1))) + r5_2

        r7_1 = self.ReLU(self.in_norm_10_1(self.conv_10_1(self.pad_10_1(r6_2))))
        r7_2 = self.in_norm_10_2(self.conv_10_2(self.pad_10_2(r7_1))) + r6_2

        r8_1 = self.ReLU(self.in_norm_11_1(self.conv_11_1(self.pad_11_1(r7_2))))
        r8_2 = self.in_norm_11_2(self.conv_11_2(self.pad_11_2(r8_1))) + r7_2

        # Up Convolution Blocks
        u1 = self.ReLU(self.in_norm_12_1(self.deconv_12_2(self.deconv_12_1(r8_2))))
        u2 = self.ReLU(self.in_norm_13_1(self.deconv_13_2(self.deconv_13_1(u1))))

        # Final Block
        y = self.Tanh(self.conv_14_1(self.pad_14_1(u2)))

        return y


class Discriminator(nn.Module):
    def __init__(self, init_weights_path=None):
        super(Discriminator, self).__init__()

        # Activation Functions
        self.LeakyReLU = nn.LeakyReLU(0.2, True)
        self.Sigmoid = nn.Sigmoid()

        ## 1st Block
        self.conv_01_1 = nn.Conv2d(3, 32, 3, 1, 1)
        # LeakyReLU

        ## 2nd Block
        self.conv_02_1 = nn.Conv2d(32, 64, 3, 2, 1)
        # LeakyReLU
        self.conv_02_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.in_norm_02_1 = InstanceNormalization(128)
        # LeakyReLU

        ## 3rd Block
        self.conv_03_1 = nn.Conv2d(128, 128, 3, 2, 1)
        # Leaky ReLU
        self.conv_03_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.in_norm_03_1 = InstanceNormalization(256)
        # LeakyReLU

        ## 4th Block
        self.conv_04_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in_norm_04_1 = InstanceNormalization(256)
        # LeakyReLU

        ## Final Block
        self.conv_05_1 = nn.Conv2d(256, 1, 3, 1, 1)
        # Sigmoid

        # Initialize weights
        if init_weights_path is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(init_weights_path)["model_state_dict"])

    def forward(self, x):
        d1 = self.LeakyReLU(self.conv_01_1(x))
        d2 = self.LeakyReLU(self.in_norm_02_1(self.conv_02_2(self.LeakyReLU(self.conv_02_1(d1)))))
        d3 = self.LeakyReLU(self.in_norm_03_1(self.conv_03_2(self.LeakyReLU(self.conv_03_1(d2)))))
        d4 = self.LeakyReLU(self.in_norm_04_1(self.conv_04_1(d3)))
        d5 = self.Sigmoid(self.conv_05_1(d4))

        return d5


class VGG19(nn.Module):
    def __init__(self, init_weights_path=None):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        # Initialize weights
        if init_weights_path is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(init_weights_path)["model_state_dict"])

    def forward(self, x):
        y = self.features(x)

        return y


class Inception_v3(nn.Module):
    def __init__(self, init_weights_path=None):
        super(Inception_v3, self).__init__()

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # Initialize weights
        if init_weights_path is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(init_weights_path))

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]

        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]

        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)


class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def __call__(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


if __name__ == "__main__":
    avatar = Inception_v3()
    origin = Inception_v3()

    def sanity_check(src_state_dict, tgt_state_dict):
        print("======== Sanity Check ========")

        ret = True

        for key, val in src_state_dict.items():
            if key in tgt_state_dict.keys():
                if val.size() == tgt_state_dict[key].size():
                    pass
                else:
                    print("The key is same. But Val size is Diff.")
                    print("src key:", key)
                    print("src size:", key.size())
                    print("dst key:", key)
                    print("dst size:", tgt_state_dict[key].size())
                    ret = False
            else:
                print("The key is not in original model state dict.")
                print("src key:", key)
                print("src size:", key.size())
                ret = False

        return ret

    avatar_state_dict = avatar.state_dict()
    origin_state_dict = origin.state_dict()

    # sanity check
    if sanity_check(avatar_state_dict, origin_state_dict):
        print("Sanity Check passed")
    else:
        print("Sanity Check not passed.")

    # Model Weight Save.
    update_state_dict = {key: val for key, val in origin_state_dict.items() if key in avatar_state_dict}
    avatar_state_dict.update(update_state_dict)

    avatar.load_state_dict(avatar_state_dict)

    torch.save(avatar.state_dict(), "./Saved_model/pretrained_inception_v3.pth")
