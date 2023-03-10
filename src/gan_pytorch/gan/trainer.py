from pathlib import Path

import numpy as np
import torch.nn
import torchvision
from loguru import logger
from torch.autograd import Variable
from torch.utils.data import DataLoader

from gan_pytorch.datasets.core import SimpleDataset

from .model import Discriminator, Generator


class Trainer(object):
    def __init__(
        self,
        _z_dim,
        shape,
        channel,
        g_lr,
        d_lr,
        adam_beta_1,
        adam_beta_2,
        epochs,
        batch_size,
        output_dir,
    ):
        super(Trainer, self).__init__()
        self.z_dim = _z_dim
        self.h = shape[0]
        self.w = shape[1]
        self.c = channel
        self.output_dir = output_dir
        self.g = Generator(self.z_dim, (self.h, self.w), self.c)
        self.d = Discriminator((self.h, self.w), self.c)
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epochs = epochs
        self.batch_size = batch_size
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

        self.data_loader = DataLoader(
            dataset=SimpleDataset(r"D:\data\fg", include_patterns=["*.bmp"]),
            # torchvision.datasets.MNIST(
            #     "../../../tests/test_data",
            #     train=True,
            #     download=True,
            #     transform=torchvision.transforms.Compose(
            #         [
            #             torchvision.transforms.ToTensor(),
            #             torchvision.transforms.Normalize((0.5,), (0.5,)),
            #         ]
            #     ),
            # ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def run(self):
        for epoch in range(self.epochs):
            for idx, (imgs, _) in enumerate(self.data_loader):
                if self.cuda:
                    real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())
                    valid = Variable(
                        torch.ones(self.batch_size, 1).cuda(), requires_grad=False
                    )
                    fake = Variable(
                        torch.zeros(self.batch_size, 1).cuda(), requires_grad=False
                    )
                    z = Variable(
                        torch.FloatTensor(
                            np.random.normal(0, 1, (self.batch_size, self.z_dim))
                        ).cuda()
                    )
                else:
                    real_imgs = Variable(imgs.type(torch.FloatTensor))
                    valid = Variable(
                        torch.ones(self.batch_size, 1), requires_grad=False
                    )
                    fake = Variable(
                        torch.zeros(self.batch_size, 1), requires_grad=False
                    )
                    z = Variable(
                        torch.FloatTensor(
                            np.random.normal(0, 1, (self.batch_size, self.z_dim))
                        )
                    )

                self.opt_g.zero_grad()
                g_outs = self.g(z)
                d_preds_by_g_outs = self.d(g_outs)
                g_loss = self.loss(d_preds_by_g_outs, valid)
                g_loss.backward()
                self.opt_g.step()

                self.opt_d.zero_grad()

                d_preds_by_real = self.d(real_imgs)
                d_preds_by_g_outs = self.d(g_outs.detach())
                d_loss = self.loss(d_preds_by_real, valid) + self.loss(
                    d_preds_by_g_outs, fake
                )
                d_loss.backward()
                self.opt_d.step()

                if self.cuda:
                    pass
                else:
                    logger.info(f"G loss: {g_loss.data}")
                    logger.info(f"D loss: {d_loss.data}")
                    logger.info(
                        f"iter {idx}/{len(self.data_loader)}, epoch {epoch+1}/{self.epochs}"
                    )

            if epoch % 100 == 0:
                torchvision.utils.save_image(
                    g_outs[:25],
                    f"{self.output_dir}/{epoch}.png",
                    nrow=5,
                    normalize=True,
                    range=(-1, 1),
                )
                torch.save(self.g, f"{self.output_dir}/g_{epoch}.pth")
                torch.save(self.d, f"{self.output_dir}/d_{epoch}.pth")
