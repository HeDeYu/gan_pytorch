from pathlib import Path

import numpy as np
import torch.nn
import torchvision
from loguru import logger

from .model import Discriminator, Generator, init_weights_normal


class Trainer(object):
    def __init__(
        self,
        num_cls,
        z_dim,
        channel,
        img_size,
        g_lr,
        d_lr,
        adam_beta_1,
        adam_beta_2,
        epochs,
        batch_size,
        output_dir,
        g_base_channel,
        d_base_channel,
    ):
        super(Trainer, self).__init__()
        self.num_cls = num_cls
        self.z_dim = z_dim
        self.channel = channel
        self.output_dir = output_dir
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.g = Generator(num_cls, z_dim, g_base_channel, channel)
        self.d = Discriminator(num_cls, d_base_channel, channel)
        self.img_size = img_size
        self.loss = torch.nn.BCELoss()
        self.cuda = False
        # if torch.cuda.is_available():
        #     logger.info("use cuda")
        #     self.g.cuda()
        #     self.d.cuda()
        #     self.loss.cuda()
        # else:
        #     raise NotImplemented

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.g.apply(init_weights_normal)
        self.d.apply(init_weights_normal)

        self.opt_g = torch.optim.Adam(
            self.g.parameters(),
            lr=self.g_lr,
            betas=(self.adam_beta_1, self.adam_beta_2),
        )
        self.opt_d = torch.optim.Adam(
            self.d.parameters(),
            lr=self.d_lr,
            betas=(self.adam_beta_1, self.adam_beta_2),
        )

        dataset = torchvision.datasets.MNIST(
            "../../../tests/test_data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(self.img_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

        self.data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def run(self):
        for epoch in range(self.epochs):
            for batch_idx, (imgs, labels) in enumerate(self.data_loader):

                if self.cuda:
                    valid = torch.autograd.Variable(
                        torch.ones(self.batch_size).cuda(), requires_grad=False
                    )
                    fake = torch.autograd.Variable(
                        torch.zeros(self.batch_size).cuda(), requires_grad=False
                    )
                    real_imgs = torch.autograd.Variable(
                        imgs.type(torch.FloatTensor).cuda()
                    )
                    z = torch.autograd.Variable(
                        torch.randn((self.batch_size, self.z_dim, 1, 1)).cuda()
                    )
                    gen_labels = (torch.rand(self.batch_size, 1) * self.num_cls).type(
                        torch.LongTensor
                    )
                    y = torch.zeros(self.batch_size, self.num_cls)
                    y = torch.autograd.Variable(
                        y.scatter_(1, gen_labels.view(self.batch_size, 1), 1)
                        .view(self.batch_size, self.num_cls, 1, 1)
                        .cuda()
                    )

                else:
                    valid = torch.autograd.Variable(
                        torch.ones(self.batch_size), requires_grad=False
                    )
                    fake = torch.autograd.Variable(
                        torch.zeros(self.batch_size), requires_grad=False
                    )
                    real_imgs = torch.autograd.Variable(imgs.type(torch.FloatTensor))
                    z = torch.autograd.Variable(
                        torch.randn((self.batch_size, self.z_dim, 1, 1))
                    )
                    gen_labels = (torch.rand(self.batch_size, 1) * self.num_cls).type(
                        torch.LongTensor
                    )
                    y = torch.zeros(self.batch_size, self.num_cls)
                    y = torch.autograd.Variable(
                        y.scatter_(1, gen_labels.view(self.batch_size, 1), 1).view(
                            self.batch_size, self.num_cls, 1, 1
                        )
                    )

                real_labels = torch.zeros(self.batch_size, self.num_cls)
                real_labels = (
                    real_labels.scatter_(1, labels.view(self.batch_size, 1), 1)
                    .view(
                        self.batch_size,
                        self.num_cls,
                        1,
                        1,
                    )
                    .contiguous()
                )
                real_labels_expanded = torch.autograd.Variable(
                    real_labels.expand(-1, -1, self.img_size, self.img_size)
                )

                gen_imgs = self.g(z, y)
                y_expanded = (
                    y.view(self.batch_size, self.num_cls, 1, 1)
                    .contiguous()
                    .expand(-1, -1, 32, 32)
                )

                self.opt_d.zero_grad()
                d_loss_for_real = self.loss(
                    self.d(real_imgs, real_labels_expanded).squeeze(), valid
                )
                d_loss_for_gen = self.loss(
                    self.d(gen_imgs.detach(), y_expanded).squeeze(), fake
                )
                d_loss = d_loss_for_real + d_loss_for_gen
                d_loss.backward()
                self.opt_d.step()

                self.g.zero_grad()
                g_loss = self.loss(self.d(gen_imgs, y_expanded).squeeze(), valid)
                g_loss.backward()
                self.opt_g.step()

                if self.cuda:
                    pass
                else:
                    logger.info(f"G loss: {g_loss.data}")
                    logger.info(f"D loss: {d_loss.data}")
                    logger.info(
                        f"iter {batch_idx}/{len(self.data_loader)}, epoch {epoch+1}/{self.epochs}"
                    )
            if True:
                z_ = torch.autograd.Variable(
                    torch.FloatTensor(
                        np.random.normal(0, 1, (self.num_cls**2, self.z_dim, 1, 1))
                    )
                )
                l_ = (
                    torch.LongTensor(np.array([n for n in range(self.num_cls)]))
                    .view(self.num_cls, 1)
                    .expand(-1, self.num_cls)
                    .contiguous()
                )
                y_ = torch.zeros(self.num_cls**2, self.num_cls)
                y_ = torch.autograd.Variable(
                    y_.scatter_(1, l_.view(self.num_cls**2, 1), 1).view(
                        self.num_cls**2, self.num_cls, 1, 1
                    )
                )
                g_outs = self.g(z_, y_).view(
                    -1, self.channel, self.img_size, self.img_size
                )
                torchvision.utils.save_image(
                    g_outs.data,
                    f"{self.output_dir}/{epoch}.png",
                    nrow=self.num_cls,
                    normalize=True,
                    range=(-1, 1),
                )
                torch.save(self.g, f"{self.output_dir}/g_{epoch}.pth")
                torch.save(self.d, f"{self.output_dir}/d_{epoch}.pth")
