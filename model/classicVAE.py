import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Encoder(nnx.Module):
    """_summary_
    """
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, *,  rngs: nnx.Rngs): 
        # Linear architecture, may be improved with convolutional layers ?
        self.rngs = rngs
        self.linear1 = nnx.Linear(in_features = in_dim, out_features = hidden_dim, rngs = rngs)
        self.linear_mu = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)
        self.linear_logvar = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.linear1(x))
        mu = nnx.relu(self.linear_mu(x))
        logvar = nnx.relu(self.linear_logvar(x))
        return mu, logvar




class Decoder(nnx.Module):
    """_summary_
    """
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.linear1 = nnx.Linear(in_features = latent_dim, out_features=hidden_dim, rngs = rngs)
        self.linear2 = nnx.Linear(in_features = hidden_dim, out_features = out_dim, rngs = rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = nnx.relu(self.linear1(z))
        return nnx.sigmoid(self.linear2(z))





class ClassicVAE(nnx.Module):
    """_summary_
    """
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim : int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, rngs: nnx.Rngs) -> jnp.ndarray:
        mu, log_sigma = self.encoder(x)
        z = mu + jnp.exp(log_sigma)*jr.normal(self.rngs.params(), shape = (self.latent_dim,))
        ps = self.decoder(z)
        sample = jr.bernoulli(rngs.params(), ps)
        return sample
    
    def generate(self,*, rngs: nnx.Rngs):
        z = jr.normal(rngs.params(), shape = (self.latent_dim,))
        p = self.decoder(z)
        sample = jr.bernoulli(rngs.params(), p)
        return sample


if __name__ == "__main__":
    in_dim = 784
    latent_dim = 64
    hidden_dim = 128
    rngs = nnx.Rngs(jax.random.PRNGKey(42))
    model = ClassicVAE(in_dim = in_dim, latent_dim = latent_dim, hidden_dim = hidden_dim, rngs = rngs)
    nnx.display(model)



