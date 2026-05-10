import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal

from model.classicVAE import Encoder_conv, Decoder_conv



class GMMVAE(nnx.Module):
    """Variational Autoencoder with GMM prior combining an encoder, reparameterization, and decoder."""

    def __init__(
        self, 
        in_dim: int, 
        latent_dim: int, 
        hidden_dim : int, 
        K : int, 
        rngs: nnx.Rngs, 
        beta : float = 1., 
        learn_prior : bool = True,
        ):
        """
        Args:
            in_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of hidden units in encoder and decoder.
            K (int) : Number of clusters in the GMM model.
            rngs (nnx.Rngs): PRNG keys for parameter initialization and sampling.
            beta (float) : beta parameter of beta-VAE.
            learn_prior (bool) : indicates if the prior is learned or fixed.
        """
        self.rngs = rngs
        self.latent_dim = nnx.static(latent_dim)    
        self.encoder =  Encoder_conv(in_dim=in_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.decoder = Decoder_conv(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim, rngs=rngs)
        # GMM parameters
        self.K = nnx.static(K) 
        if learn_prior :
            self.logit_alpha_gmm = nnx.Param(jnp.zeros(shape=(K,)))
            self.mu_gmm = nnx.Param(5.0 * jnp.eye(K, latent_dim))
            self.logvar_gmm = nnx.Param(jnp.zeros(shape=(K, latent_dim)))
        else :
            self.logit_alpha_gmm = nnx.Variable(jnp.zeros(shape=(K,)))
            self.mu_gmm = nnx.Variable(5.0 * jnp.eye(K, latent_dim))
            self.logvar_gmm = nnx.Variable(jnp.zeros(shape=(K, latent_dim)))
        # Beta-VAE 
        self.beta = nnx.static(beta)


    def generate(self, key : jax.Array) -> jnp.array : 
        """Generates a sample by decoding a random latent vector drawn from the prior.

        Args:
            key (jax.Array): PRNG key for sampling.

        Returns:
            jnp.array: Binary image sampled from the decoded Bernoulli distribution.
        """
        key1, key2, key3 = jr.split(key, 3)
        p = jax.nn.softmax(self.logit_alpha_gmm.value)
        cluster = jr.choice(key1, self.K, p = p)
        mu_cluster, cov_cluster = self.mu_gmm.value[cluster], jnp.diag(jnp.exp(self.logvar_gmm.value[cluster]))
        z = jr.multivariate_normal(key2, mean = mu_cluster, cov = cov_cluster)
        ps = self.decoder(z)
        return jr.bernoulli(key3, ps)

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

    def prior_pdf(self, z : jnp.array) -> jnp.array :
        """Evaluates the GMM prior density at a given latent vector.

        Args:
            z (jnp.array): Latent vector of shape (latent_dim,).

        Returns:
            jnp.array: Scalar prior density p(z) under the learned GMM.
        """
        def gaussian_pdf_single(mu_single, logvar_single): 
            cov = jnp.diag(jnp.exp(logvar_single))
            return multivariate_normal.pdf(z, mu_single, cov)
        gauss_comp = jax.vmap(gaussian_pdf_single)(self.mu_gmm.value, self.logvar_gmm.value)
        pdf = jnp.sum(jax.nn.softmax(self.logit_alpha_gmm.value)*gauss_comp)
        return pdf

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
        