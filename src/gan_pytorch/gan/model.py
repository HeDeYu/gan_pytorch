import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, _z_dim, output_shape, output_channel):
        super(Generator, self).__init__()
        self.z_dim = _z_dim
        self.h = output_shape[0]
        self.w = output_shape[1]
        self.c = output_channel
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.c * self.h * self.w),
            nn.Tanh(),
        )

    def forward(self, x):
        o = self.model(x)
        o = o.view(o.size(0), self.c, self.h, self.w)
        return o


class Discriminator(nn.Module):
    def __init__(self, input_shape, input_channel):
        super(Discriminator, self).__init__()
        self.h = input_shape[0]
        self.w = input_shape[1]
        self.c = input_channel
        self.model = nn.Sequential(
            nn.Linear(self.c * self.h * self.w, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        prob = self.model(x_flat)
        return prob
