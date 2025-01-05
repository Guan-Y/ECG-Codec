import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def MultiScale_STFT(x, ms_stft):
    assert x.dim() == 3
    _, channel, length = x.shape
    specs = []
    for stft_i in ms_stft:
        y = stft_i(x)
        y = torch.cat([y.real, y.imag], dim=1)
        specs.append(y)
    return specs


def compute_loss_t(inputs, recons):
    loss_t = F.l1_loss(inputs, recons)
    return loss_t


def compute_loss_f(inputs, recons, ms_stft):
    s_in = MultiScale_STFT(inputs, ms_stft)
    s_recons = MultiScale_STFT(recons, ms_stft)
    loss_s = 0

    for i in range(0, len(s_in)):
        # loss_s += F.mse_loss(s_in[i], s_recons[i]) + F.l1_loss(s_in[i], s_recons[i])
        loss_s += 0.5 * F.l1_loss(s_in[i], s_recons[i]) + 0.5 * F.mse_loss(s_in[i], s_recons[i])
    return loss_s

def Rate_Distortion_Loss(inputs, recons, likelihoods, lmbda=10):
    distortion = F.mse_loss(inputs, recons)
    _, _, num_points = inputs.shape
    bpp_loss = torch.log(likelihoods).sum() / (-math.log(2) * num_points)
    out_loss = lmbda * distortion + bpp_loss
    out = {
        'mse_loss': distortion,
        'bpp_loss': bpp_loss,
        'out_loss': out_loss
    }
    return out


def Rate_Loss(indices, bins):
    num_points = bins
    pdf = torch.zeros(num_points, device=indices.device)
    loss_rate = 0
    pdfs = []

    for frame in indices:
        unique_elements, counts = torch.unique(frame, return_counts=True)
        for element, count in zip(unique_elements, counts):
            pdf[int(element)] += float(count) / sum(counts)
        # pdf = [float(item) / sum(pdf) for item in pdf]
        pdfs.append(pdf)

    for pdf in pdfs:
        pdf[pdf == 0] = 1
        loss_rate = (-torch.log(pdf).sum()) / num_points

    return loss_rate


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature
