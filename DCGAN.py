from __future__ import print_function

import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import HTML


def weights_init(m):
    """
    Initialize weights per Goodfellow paper
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator
    """
    def __init__(self, ngpu_):
        super(Generator, self).__init__()
        self.ngpu = ngpu_
        self.main = nn.Sequential(
            # Initial 100
            nn.ConvTranspose2d(n_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, n_chan, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, ngpu_):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu_
        self.main = nn.Sequential(
            # nc x 64 x 64
            nn.Conv2d(n_chan, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class GAN:
    """
    Main GAN class
    """
    def __init__(self, manual_seed_, workers_, batch_size_, img_size_, n_chan_, n_z_, ngf_, ndf_, n_epochs_, lr_,
                 beta1_, ngpu_):
        """
        Initialize class variables
        """
        self.manual_seed = manual_seed_
        if manual_seed_ != 0:
            random.seed(self.manual_seed)
            torch.manual_seed(self.manual_seed)
        self.dataroot = dataroot
        self.workers = workers_
        self.batch_size = batch_size_
        self.img_size = img_size_
        self.n_chan = n_chan_
        self.n_z = n_z_
        self.ngf = ngf_
        self.ndf = ndf_
        self.n_epochs = n_epochs_
        self.lr = lr_
        self.beta1 = beta1_
        self.ngpu = ngpu_
        self.img_list = list()
        self.g_losses = list()
        self.d_losses = list()
        self.dataset = dset.ImageFolder(root=self.dataroot,
                                        transform=transforms.Compose([
                                                  transforms.Resize(img_size_),
                                                  transforms.CenterCrop(img_size_),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=self.workers)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu_ > 0) else 'cpu')
        # device = torch.device('cpu')
        self.real_batch = next(iter(self.dataloader))
        self.netG = Generator(ngpu_).to(self.device)
        if (self.device.type == 'cuda') and (ngpu_ > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu_)))
        self.netG.apply(weights_init)

        self.netD = Discriminator(ngpu_).to(self.device)
        if (self.device.type == 'cuda') and (ngpu_ > 1):
            self.netD = nn.DataParallel(self.netD, list(range(ngpu_)))
        self.netD.apply(weights_init)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, n_z_, 1, 1, device=self.device)

    def load_gen(self):
        """
        Load generator state dict
        """
        gen = torch.load('generator')
        self.netG.load_state_dict(gen)

    def save_model(self):
        """
        Save state dict for generator and discriminator
        """
        torch.save(self.netG.state_dict(), 'generator')
        torch.save(self.netD.state_dict(), 'discriminator')

    def single_test(self):
        """
        Run single random image through generator as test
        """
        out_test = self.netG(torch.randn(1, n_z, 1, 1, device=self.device)).detach().squeeze().cpu()
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid([out_test], normalize=True), (1, 2, 0)))
        plt.show()

    def train(self):
        """
        Training loop
        """
        real_label = 1
        fake_label = 0
        optimizer_d = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_g = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        iters = 0
        print('Start training')
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader):
                # Update network, train real batch
                self.netD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device, dtype=torch.float)
                # forward
                output = self.netD(real_cpu).view(-1)
                err_d_real = self.criterion(output, label)
                err_d_real.backward()
                d_x = output.mean().item()

                # All fake
                noise = torch.randn(b_size, n_z, 1, 1, device=self.device)

                fake = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fake.detach()).view(-1)
                err_d_fake = self.criterion(output, label)

                err_d_fake.backward()
                d_g_z1 = output.mean().item()
                err_d = err_d_fake + err_d_real
                optimizer_d.step()

                # update generator
                self.netG.zero_grad()
                label.fill_(real_label)

                output = self.netD(fake).view(-1)
                err_g = self.criterion(output, label)
                err_g.backward()
                d_g_z2 = output.mean().item()
                optimizer_g.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, n_epochs, i, len(self.dataloader), err_d.item(), err_d.item(), d_x, d_g_z1, d_g_z2))
                self.g_losses.append(err_g.item())
                self.d_losses.append(err_d.item())
                if (iters % 500 == 0) or ((epoch == n_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

    def post_training_eval(self):
        """
        Training evaluation loop
        """
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G")
        plt.plot(self.g_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.dataloader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),
                                (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    manual_seed = 0
    dataroot = 'celeba'
    workers = 0
    batch_size = 128
    img_size = 64
    n_chan = 3
    n_z = 100
    ngf = 64
    ndf = 64
    n_epochs = 10
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    main_gan = GAN(manual_seed, workers, batch_size, img_size, n_chan, n_z, ngf, ndf, n_epochs, lr, beta1, ngpu)
    main_gan.load_gen()
    main_gan.single_test()
