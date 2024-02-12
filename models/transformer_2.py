### based on https://github.com/BaratiLab/ROMER/blob/master/Model.py

import torch
from torch import nn
import numpy as np



class Encoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,16),
                                 nn.LeakyReLU(),
                                 nn.Linear(16,32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32,64),
                                nn.LeakyReLU(),
                                 nn.Linear(64,128)
                                 )

    def forward(self, x):

        return self.net(x)


class Decoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.net = nn.Sequential(nn.Linear(128,64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64,32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32,16),
                                 nn.LeakyReLU(),
                                 nn.Linear(16,1)
                                 )

    def forward(self, x):

        return self.net(x)




class Transformer2D_Layer(nn.Module):

    def __init__(self, embed_dim=128, num_heads=8,
                 kdim=None, vdim=None, hidden_dim=256):

        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         kdim=kdim, vdim=vdim,
                                         batch_first=True)
        self.FF = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.MHA(x, x, x, need_weights=False)[0]
        x = x + self.FF(x)
        return x


class Transformer2D(nn.Module):

    def __init__(self, shape=(4, 4), n_layers=6,
                 MHA_kwargs=dict(embed_dim=128, num_heads=8,
                                 kdim=None, vdim=None, hidden_dim=256),
                 periodic=True):

        super().__init__()
        self.spatial_shape = shape  # (nx, ny)
        nx, ny = shape

        if periodic:
            x, y = torch.meshgrid(torch.arange(nx), torch.arange(ny))
            x_freq = torch.fft.rfftfreq(nx)[1:, None, None]
            y_freq = torch.fft.rfftfreq(ny)[1:, None, None]
            x_sin = torch.sin(2*np.pi*x_freq*x)
            x_cos = torch.cos(2*np.pi*x_freq*x)
            y_sin = torch.sin(2*np.pi*y_freq*y)
            y_cos = torch.cos(2*np.pi*y_freq*y)
            pos_info = torch.cat([x_sin, x_cos, y_sin, y_cos])
        else:
            x, y = torch.meshgrid(torch.arange(1, nx+1)/nx,
                                  torch.arange(1, ny+1)/ny)
            pos_info = torch.stack([x, y])

        dim_pos = pos_info.shape[0]
        self.pos_info = pos_info.unsqueeze(0) # for the batch dimension

        self.pos_embedder = nn.Sequential(
            nn.Conv2d(dim_pos, dim_pos*4, 1), nn.LeakyReLU(),
            nn.Conv2d(dim_pos*4, MHA_kwargs['embed_dim'], 1)
            )

        layers = [Transformer2D_Layer(**MHA_kwargs) for i in range(n_layers)]
        self.transformer = nn.Sequential(*layers)

    def forward(self, x):

        x += self.pos_embedder(self.pos_info.to(x.device))
        x = x.flatten(-2)
        x = self.transformer(x)
        x = x.reshape(*x.shape[:-1], *self.spatial_shape)
        return x