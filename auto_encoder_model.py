import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # in -> 32x32x3
            nn.Conv2d(3, 8, 4, stride=2),
            nn.LeakyReLU(),  # 15x15x8
            nn.Conv2d(8, 16, 3, stride=2),
            nn.LeakyReLU(),  # 7x7x16
            nn.Conv2d(16, 32, 3, stride=2),
            nn.LeakyReLU(),  # 3x3x32
            nn.Conv2d(32, 64, 3, stride=2),
            nn.Tanh(),  # 1x1x64
        )
        self.decoder = nn.Sequential(
            # in -> 1x1x64
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.LeakyReLU(),  # 3x3x32
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.LeakyReLU(),  # 7x7x16
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.LeakyReLU(),  # 15x15x8
            nn.ConvTranspose2d(8, 3, 4, stride=2),
            nn.Sigmoid(),  # 32x32x3
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
