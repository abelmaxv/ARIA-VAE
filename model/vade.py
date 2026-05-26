import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal

from model.classicVAE import Encoder_conv, Decoder_conv


class VaDE(nnx.Module):
    """VaDE : Variational autoencoder with GMM prior and analytical categorical posterior.
    """

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

    def generate_mean(self, key: jax.Array) -> jnp.array:
        """Generates the decoder mean (Bernoulli probabilities) from the prior.

        Returns a smooth grey-scale image — the expected pixel values — rather
        than a noisy binary sample.  Prefer this for visualisation.

        Args:
            key (jax.Array): PRNG key for latent sampling.

        Returns:
            jnp.array: Decoder probabilities in [0, 1] of shape (in_dim,).
        """
        key1, key2 = jr.split(key)
        p = jax.nn.softmax(self.logit_alpha_gmm.value)
        cluster = jr.choice(key1, self.K, p=p)
        mu_cluster = self.mu_gmm.value[cluster]
        cov_cluster = jnp.diag(jnp.exp(self.logvar_gmm.value[cluster]))
        z = jr.multivariate_normal(key2, mean=mu_cluster, cov=cov_cluster)
        return self.decoder(z)

    def generate_from_component(self, key: jax.Array, component: int) -> jnp.array:
        """Generates the decoder mean from a specific GMM component.

        Useful for class-conditional visualisation: each row of a grid can be
        sampled from one component, reproducing the style of Figure 3(d) in the
        VaDE paper.

        Args:
            key (jax.Array): PRNG key for latent sampling.
            component (int): GMM component index in [0, K).

        Returns:
            jnp.array: Decoder probabilities in [0, 1] of shape (in_dim,).
        """
        mu_k = self.mu_gmm.value[component]
        cov_k = jnp.diag(jnp.exp(self.logvar_gmm.value[component]))
        z = jr.multivariate_normal(key, mean=mu_k, cov=cov_k)
        return self.decoder(z)

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

    def encode_class(self, x : jnp.array) -> int:
        """Return the most probable cluster index for an input.

        Args:
            x (jnp.array): Input Tensor

        Returns:
            int : predicted cluster index
        """
        return int(self.class_prob(x).argmax())

    def class_prob(self, x : jnp.array) -> jnp.array :
        """Compute the analytical posterior over clusters q(c | x).

        Args:
            x (jnp.array): Input tensor of shape (in_dim,).

        Returns:
            jnp.array: Cluster probabilities of shape (K,), sums to 1.
        """
        mu_enc, _ = self.encoder(x)  # only the mean is used

        def log_component(mu_j, logvar_j):
            cov_j = jnp.diag(jnp.exp(logvar_j))
            return multivariate_normal.logpdf(mu_enc, mu_j, cov_j)

        log_gauss = jax.vmap(log_component)(
            self.mu_gmm.value, self.logvar_gmm.value
        )  # (K,)
        log_alpha = jax.nn.log_softmax(self.logit_alpha_gmm.value)  # (K,)
        log_q_c = jax.nn.log_softmax(log_alpha + log_gauss)         # (K,)
        return jnp.exp(log_q_c)

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
        mu, logvar = self.encoder(x)
        key1, key2 = jr.split(key)
        z = mu + jnp.exp(0.5 * logvar) * jr.normal(key1, shape=(self.latent_dim,))
        return self.decoder(z)
        