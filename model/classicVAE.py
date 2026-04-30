import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



class Encoder_linear(nnx.Module):
    """Maps input data to a distribution in latent space via a linear architecture."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, *,  rngs: nnx.Rngs):
        """
        Args:
            in_dim (int): Dimensionality of the input.
            hidden_dim (int): Number of hidden units.
            latent_dim (int): Dimensionality of the latent space.
            rngs (nnx.Rngs): PRNG keys for parameter initialization.
        """
        # Linear architecture, may be improved with convolutional layers ?
        self.rngs = rngs
        self.linear1 = nnx.Linear(in_features = in_dim, out_features = hidden_dim, rngs = rngs)
        self.linear_mu = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)
        self.linear_logvar = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encodes input into mean and log-variance of the latent distribution.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Mean and log-variance of the latent distribution.
        """
        x_hidden = nnx.relu(self.linear1(x))
        mu = self.linear_mu(x_hidden)
        logvar = self.linear_logvar(x_hidden)
        return mu, logvar


class Encoder_conv(nnx.Module):
    """Maps input data to a distribution in latent space via a convolutional architecture."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, *,  rngs: nnx.Rngs):
        """
        Args:
            in_dim (int): Dimensionality of the input.
            hidden_dim (int): Number of hidden units.
            latent_dim (int): Dimensionality of the latent space.
            rngs (nnx.Rngs): PRNG keys for parameter initialization.
        """
        self.rngs = rngs
        self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",  rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.linear = nnx.Linear(in_features=4*4*128, out_features=hidden_dim, rngs=rngs)
        self.linear_mu = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)
        self.linear_logvar = nnx.Linear(in_features = hidden_dim, out_features = latent_dim, rngs = rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encodes input into mean and log-variance of the latent distribution.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Mean and log-variance of the latent distribution.
        """
        x = x.reshape(1, 28, 28, 1)
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape(-1)
        x = nnx.relu(self.linear(x))
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar



class Decoder_linear(nnx.Module):
    """Maps a latent vector back to the output space via a linear architecture."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        """
        Args:
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units.
            out_dim (int): Dimensionality of the reconstructed output.
            rngs (nnx.Rngs): PRNG keys for parameter initialization.
        """
        self.rngs = rngs
        self.linear1 = nnx.Linear(in_features = latent_dim, out_features=hidden_dim, rngs = rngs)
        self.linear2 = nnx.Linear(in_features = hidden_dim, out_features = out_dim, rngs = rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decodes a latent vector into Bernoulli probabilities for the output.

        Args:
            z (jnp.ndarray): Latent vector.

        Returns:
            jnp.ndarray: Output tensor with values in (0, 1).
        """
        z_hidden = nnx.relu(self.linear1(z))
        return nnx.sigmoid(self.linear2(z_hidden))


class Decoder_conv(nnx.Module):
    """Maps a latent vector back to the output space via a convolutional architecture."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        """
        Args:
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units.
            out_dim (int): Dimensionality of the reconstructed output.
            rngs (nnx.Rngs): PRNG keys for parameter initialization.
        """
        self.rngs = rngs
        self.linear1 = nnx.Linear(in_features = latent_dim, out_features=hidden_dim, rngs = rngs)
        self.linear2 = nnx.Linear(in_features = hidden_dim, out_features = out_dim, rngs = rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decodes a latent vector into Bernoulli probabilities for the output.

        Args:
            z (jnp.ndarray): Latent vector.

        Returns:
            jnp.ndarray: Output tensor with values in (0, 1).
        """
        z_hidden = nnx.relu(self.linear1(z))
        return nnx.sigmoid(self.linear2(z_hidden))





class ClassicVAE(nnx.Module):
    """Variational Autoencoder combining an encoder, reparameterization, and decoder."""

    def __init__(self, in_dim: int, latent_dim: int, hidden_dim : int, rngs: nnx.Rngs, arch : str = "conv"):
        """
        Args:
            in_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units in encoder and decoder.
            rngs (nnx.Rngs): PRNG keys for parameter initialization and sampling.
            arch (str) : Type of architecture for the encoder ("conv" or "linear")
        """
        self.rngs = rngs
        self.latent_dim = latent_dim
        if arch == "linear" :
            self.encoder = Encoder_linear(in_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, rngs=rngs)
            self.decoder = Decoder_linear(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim, rngs=rngs)
        elif arch == "conv" : 
            self.encoder =  Encoder_conv(in_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, rngs=rngs)
            self.decoder = Decoder_conv(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim, rngs=rngs)
        else : 
            raise ValueError(f"Invalid architecture argument: '{arch}'. Must be 'conv' or 'linear'.")

    def generate(self, key : jax.Array) -> jnp.array : 
        """Generates a sample by decoding a random latent vector drawn from the prior.

        Args:
            key (jax.Array): PRNG key for sampling.

        Returns:
            jnp.array: Binary image sampled from the decoded Bernoulli distribution.
        """
        key_z, key_b = jr.split(key)
        z = jr.normal(key_z, shape=(self.latent_dim,))
        ps = self.decoder(z)
        return jr.bernoulli(key_b, ps)

    def encode(self, x : jnp.array, key : jax.Array) -> jnp.array :
        """Encodes an input and samples a latent vector from the posterior.

        Args:
            x (jnp.array): Input tensor.
            key (jax.Array): PRNG key for sampling.

        Returns:
            jnp.array: Latent vector sampled from q(z|x).
        """
        mu, logvar = self.encoder(x)
        z = jr.multivariate_normal(key, mu, jnp.diag(jnp.exp(logvar)))
        return z

    def __call__(self, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """
        Encodes input, samples a latent vector, and decodes it into a binary sample.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Bernoulli sample from the decoded output probabilities.
        """
        mu, log_sigma = self.encoder(x)
        key1, key2 = jr.split(key)
        z = mu + jnp.exp(log_sigma) * jr.normal(key1, shape=(self.latent_dim,))
        ps = self.decoder(z)
        return jr.bernoulli(key2, ps)
        




