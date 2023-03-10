import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_cls, z_dim=100, base_channel=128, output_channel=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_cls = num_cls
        d = base_channel
        self.deconv1_z = nn.ConvTranspose2d(self.z_dim, d * 8, 4, 1, 0)
        self.deconv1_bn_z = nn.BatchNorm2d(d * 8)
        self.deconv1_y = nn.ConvTranspose2d(self.num_cls, d * 8, 4, 1, 0)
        self.deconv1_bn_y = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 16, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 4)
        self.deconv4 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 2)
        self.deconv5 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d)
        self.deconv6 = nn.ConvTranspose2d(d, output_channel, 4, 2, 1)

    def forward(self, noise, condition):
        z = F.relu(self.deconv1_bn_z(self.deconv1_z(noise)))
        y = F.relu(self.deconv1_bn_z(self.deconv1_z(condition)))
        x = torch.cat([z, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.tanh(self.deconv6(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, num_cls, base_channel=128, input_channel=3):
        super(Discriminator, self).__init__()
        d = base_channel
        self.conv1_i = nn.Conv2d(input_channel, d // 2, 4, 2, 1)
        self.conv1_y = nn.Conv2d(num_cls, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def forward(self, img, condition):
        i = F.leaky_relu(self.conv1_i(img), 0.2)
        y = F.leaky_relu(self.conv1_y(condition), 0.2)
        x = torch.cat([i, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x
