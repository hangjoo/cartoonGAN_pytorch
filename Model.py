import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import init_weights, print_model


# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, ... )

class Generator(nn.Module):
    def __init__(self, weight_PATH=None):
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
        if weight_PATH is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(weight_PATH))

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
    def __init__(self, weight_PATH=None):
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
        if weight_PATH is None:
            init_weights(self)
        else:
            self.load_state_dict(torch.load(weight_PATH))

    def forward(self, x):
        d1 = self.LeakyReLU(self.conv_01_1(x))
        d2 = self.LeakyReLU(self.in_norm_02_1(self.conv_02_2(self.LeakyReLU(self.conv_02_1(d1)))))
        d3 = self.LeakyReLU(self.in_norm_03_1(self.conv_03_2(self.LeakyReLU(self.conv_03_1(d2)))))
        d4 = self.LeakyReLU(self.in_norm_04_1(self.conv_04_1(d3)))
        d5 = self.Sigmoid(self.conv_05_1(d4))

        return d5


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


class VGG19(nn.Module):
    def __init__(self, weight_PATH=None):
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
            nn.ReLU(inplace=True)
        )

        if weight_PATH is not None:
            self.load_state_dict(torch.load(weight_PATH))

    def forward(self, x):
        y = self.features(x)

        return y


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    P_VGG19 = VGG19()
    print_model(G)
    print("\n" + "=" * 90 + "\n")
    print_model(D)
    print("\n" + "=" * 90 + "\n")
    print_model(P_VGG19)
