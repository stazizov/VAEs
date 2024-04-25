from typing import Tuple, Optional, Union, List
from math import sqrt, log

import torch
from torch import nn
from torch.distributions import Normal


_size = Union[torch.Size, List[int], Tuple[int, ...]]


class Unflatten(nn.Module):
    def __init__(self, unflattened_size: _size) -> None:
        super().__init__()

        self.unflattened_size = unflattened_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *self.unflattened_size)


class VAE(nn.Module):
    '''
        Encoder and Decoder are specified for CIFAR-10/CIFAR-100 data
    '''
    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 256,
                 use_clamping: bool = False,
                 log_std_min: float = -2,
                 log_std_max: float = 5) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.hidden_dim = 256 * 26 * 26
        self.latent_dim = latent_dim
        
        # for numerical stability (just in case)
        self.use_clamping = use_clamping
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_diff = log_std_max - log_std_min

        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_std = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            # nn.Unflatten(1, (256, 26, 26)),
            Unflatten((256, 26, 26)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3),
            # there is an assumption to [0, 1] normalization (as always)
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(layer: nn.Module):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        
        mu = self.mu(z)
        log_std = self.log_std(z)

        if self.use_clamping:
            log_std = log_std.clamp(self.log_std_min, self.log_std_max)

            # alternative with soft clamping
            # log_std = self.log_std_min + (torch.tanh(log_std) + 1) * self.log_std_diff / 2
        
        std = torch.exp(log_std)
        return mu, std
    
    def decode(self,
               z: Optional[torch.Tensor] = None,
               device: str = "cpu") -> torch.Tensor:
        if z is None:
            z = torch.randn((1, self.latent_dim)).to(device)
        
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(x)
        z = mean + std * torch.randn_like(std)

        return self.decode(z), mean, std
    
    def importance_sampling_loss(self,
                                 x: torch.Tensor,
                                 beta: float = 1.0,
                                 num_samples: int = 10) -> torch.Tensor:
        # https://arxiv.org/abs/1906.02691
        # https://arxiv.org/abs/1401.4082

        # img shape: [batch_size, 3, 32, 32]
        # latent objects shape: [batch_size, latent_dim]
        batch_size, channel_dim, height, width = x.shape
        mean, std = self.encode(x)

        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)

        xx = x.repeat(num_samples, 1, 1, 1)
        mean_decoded = self.decoder(z)
        scale_factor = sqrt(beta) / 2.0

        log_prob_q_zx = Normal(loc=mean, scale=std).log_prob(z)
        
        mean_prior = torch.zeros_like(z).to(x.device)
        std_prior = torch.ones_like(z).to(x.device)
        
        log_prob_p_z = Normal(loc=mean_prior, scale=std_prior).log_prob(z)
        std_decoded = torch.ones_like(mean_decoded).to(x.device) * scale_factor
        
        log_prob_p_xz = Normal(loc=mean_decoded, scale=std_decoded).log_prob(xx)
        # maybe at this point we should change from sum to mean idk
        log_prob_p_xz = log_prob_p_xz.sum(dim=(2, 3)).view(-1, num_samples, channel_dim)

        w = log_prob_p_xz.sum(-1) + log_prob_p_z.sum(-1) - log_prob_q_zx.sum(-1)
        score = w.logsumexp(dim=-1) - log(num_samples)
        return -score

    def elbo_loss(self,
                  x: torch.Tensor,
                  beta: float = 1.0,
                  num_samples: int = 10) -> torch.Tensor:
        # img shape: [batch_size, 3, 32, 32]
        # latent objects shape: [batch_size, latent_dim]
        mean, std = self.encode(x)

        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)

        xx = x.repeat(num_samples, 1, 1, 1)
        decoded = self.decode(z)
        reconstruction_loss = (decoded - xx).pow(2).mean(dim=(1, 2, 3))

        kl_loss = -1 / 2 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        loss = reconstruction_loss + beta * kl_loss.view(-1)
        return loss



if __name__ == "__main__":
    test = torch.rand(7, 3, 32, 32)

    vae = VAE(3, 256, False)

    decoded, mean, std = vae(test)
    print(decoded.shape, mean.shape, std.shape)

    print(vae.importance_sampling_loss(test).shape)
    print(vae.elbo_loss(test).shape)
