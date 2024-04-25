from typing import Tuple, Union, List, Sequence
from math import sqrt, log

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal
from distrax import Normal


_size = Union[Sequence[int], List[int], Tuple[int, ...]]


class Unflatten(nn.Module):
    unflattened_size: _size = (256, 26, 26)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.reshape(x, (-1, *self.unflattened_size))


class VAE(nn.Module):
    in_channels: int = 3
    latent_dim: int = 256
    use_clamping: bool = False
    log_std_min: float = -2
    log_std_max: float = 5

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        encoder = nn.Sequential([
            nn.Conv(features=64, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            nn.relu,
            nn.Conv(features=128, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            nn.relu
        ])
        # 256 * 26 * 26
        
        mu_net = nn.Dense(features=self.latent_dim, kernel_init=orthogonal(sqrt(2)))
        log_std_net = nn.Dense(features=self.latent_dim, kernel_init=orthogonal(sqrt(2)))
        
        decoder = nn.Sequential([
            nn.Dense(features=256*26*26, kernel_init=orthogonal(sqrt(2))),
            Unflatten(),
            nn.relu,
            nn.ConvTranspose(features=128, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            nn.relu,
            nn.ConvTranspose(features=64, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            nn.relu,
            nn.ConvTranspose(features=self.in_channels, kernel_size=(3, 3), kernel_init=orthogonal(sqrt(2))),
            # there is an assumption to [0, 1] normalization (as always)
            nn.sigmoid
        ])

        z = encoder(x)  
        mu, log_std = mu_net(z), log_std_net(z)
        
        if self.use_clamping:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

            # alternative with soft clamping
            # log_std = self.log_std_min + (jnp.tanh(log_std) + 1) * (self.log_std_max - self.log_std_min) / 2
        std = jnp.exp(log_std)

        dist = Normal(mu, std)
        z = dist.sample()
        pass


def update_by_importance_sampling():
    pass


def update_by_elbo():
    pass