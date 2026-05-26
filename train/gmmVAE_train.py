import sys
from pathlib import Path

# # Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.gmmVAE import GMMVAE

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2




class GMMVAE_trainer:

    def __init__(self, model: GMMVAE, optimizer: nnx.Optimizer):
        """
        Args:
            model (GMMVAE): The VAE model to train.
            optimizer (nnx.Optimizer): Optimizer wrapping the model parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.metrics = nnx.MultiMetric(loss = nnx.metrics.Average(argname='loss'))


    def compute_loss(self, model : GMMVAE, batch : jnp.array, key : jax.Array)->jnp.array:
        """Computes the negative ELBO loss for a batch.

        Args:
            model (GMMVAE): The VAE model.
            batch (jnp.array): Batch of input images of shape (N, in_dim).
            key (jax.Array): PRNG key for sampling.

        Returns:
            jnp.array: Scalar loss value (negative ELBO).
        """
        return _compute_loss_gmm(model, batch, key)



    def train_step(self, model : GMMVAE, optimizer : nnx.Optimizer, metrics : nnx.MultiMetric, batch : jnp.array):
        """Runs one JIT-compiled training step: computes gradients, updates parameters, and logs the loss.

        Args:
            model (GMMVAE): The VAE model.
            optimizer (nnx.Optimizer): Optimizer to apply gradients.
            metrics (nnx.MultiMetric): Metrics accumulator.
            batch (jnp.array): Batch of input images of shape (N, in_dim).
        """
        _train_step_gmm(model, optimizer, metrics, batch)


    def eval_step(self, model : GMMVAE, metrics : nnx.MultiMetric, batch : jnp.ndarray) -> jnp.ndarray:
        """Runs one JIT-compiled evaluation step: computes the loss and logs it without updating parameters.

        Args:
            model (GMMVAE): The VAE model.
            metrics (nnx.MultiMetric): Metrics accumulator.
            batch (jnp.ndarray): Batch of input images of shape (N, in_dim).
        """
        _eval_step_gmm(model, metrics, batch)


    def train(self, epochs : int , train_data_loader : DataLoader, test_dataloader : DataLoader, eval_every : int = 50) -> tuple[list, list]:
        """Trains the model for a given number of epochs and returns loss histories.

        Args:
            epochs (int): Number of training epochs.
            train_data_loader (DataLoader): DataLoader for the training set.
            test_dataloader (DataLoader): DataLoader for the evaluation set.
            eval_every (int): Number of batches between evaluation steps.

        Returns:
            tuple[list, list]: Train loss history and eval loss history.
        """
        return self._run_loop(epochs, train_data_loader, test_dataloader,
                              eval_every, desc='Fine-tuning (GMM-VAE)',
                              train_fn=_train_step_gmm,
                              eval_fn=_eval_step_gmm)



    def pretrain(self, epochs: int, train_data_loader: DataLoader,
                 test_dataloader: DataLoader, eval_every: int = 50) -> tuple[list, list]:
        """Phase 1: pre-train encoder/decoder with a standard Gaussian VAE ELBO.

        Args:
            epochs (int): Number of pre-training epochs.
            train_data_loader (DataLoader): Training set loader.
            test_dataloader (DataLoader): Evaluation set loader.
            eval_every (int): Batches between evaluation steps.

        Returns:
            tuple[list, list]: Train and eval loss histories.
        """
        return self._run_loop(epochs, train_data_loader, test_dataloader,
                              eval_every, desc='Pre-training (VAE warmup)',
                              train_fn=_train_step_vae_warmup,
                              eval_fn=_eval_step_vae_warmup)


    def init_gmm_kmeans(self, train_data_loader: DataLoader, seed: int = 42) -> np.ndarray:
        """Phase 2: initialise GMM parameters with K-means on the encoded training set.

        Args:
            train_data_loader (DataLoader): Training set loader.
            seed (int): Random seed for K-means initialisation.

        Returns:
            np.ndarray: Cluster label for every training point, shape (N,).
        """
        print("K-means initialisation: encoding training set …")
        self.model.eval()

        mu_list = []
        for batch_data in tqdm(train_data_loader, desc="Encoding", leave=False):
            x = batch_data[0]
            mu_batch = jax.vmap(lambda xi: self.model.encoder(xi)[0])(x)
            mu_list.append(np.array(mu_batch))

        all_mu = np.concatenate(mu_list, axis=0)          # (N, latent_dim)
        K      = self.model.K
        D      = all_mu.shape[1]

        print(f"Running K-means with K={K} on {len(all_mu)} points …")
        centroids, labels = kmeans2(all_mu, K, minit='points', iter=100, seed=seed)

        # Per-cluster log-variance and mixing weights
        logvar_init  = np.zeros((K, D))
        counts       = np.zeros(K)

        for k in range(K):
            mask = labels == k
            counts[k] = mask.sum()
            if counts[k] > 1:
                var = all_mu[mask].var(axis=0)
                logvar_init[k] = np.log(np.maximum(var, 1e-6))
            # else: leave at 0 (unit variance)

        alpha_init = counts / counts.sum()                 # normalised weights

        # Assign to model parameters (works for Param and Variable alike)
        self.model.mu_gmm.value          = jnp.array(centroids)
        self.model.logvar_gmm.value      = jnp.array(logvar_init)
        self.model.logit_alpha_gmm.value = jnp.log(jnp.array(alpha_init) + 1e-8)

        print(f"K-means done — cluster sizes: {counts.astype(int).tolist()}")
        return labels


    def _run_loop(self, epochs, train_data_loader, test_dataloader,
                  eval_every, desc, train_fn, eval_fn):
        train_history = []
        eval_history  = []

        epoch_bar = tqdm(range(epochs), desc=desc)
        with jax.debug_nans(False):
            for epoch in epoch_bar:
                batch_bar = tqdm(train_data_loader, desc=f"Epoch {epoch}", leave=False)

                for batch_id, batch_data in enumerate(batch_bar):
                    self.model.train()
                    train_fn(self.model, self.optimizer, self.metrics, batch_data[0])

                    if batch_id > 0 and (batch_id % eval_every == 0
                                         or batch_id == len(train_data_loader) - 1):
                        train_metrics = self.metrics.compute()
                        current_train_loss = float(train_metrics["loss"])
                        train_history.append(current_train_loss)
                        self.metrics.reset()

                        self.model.eval()
                        for _, test_batch_data in enumerate(test_dataloader):
                            eval_fn(self.model, self.metrics, test_batch_data[0])

                        eval_metrics = self.metrics.compute()
                        current_eval_loss = float(eval_metrics["loss"])
                        eval_history.append(current_eval_loss)
                        self.metrics.reset()

                        batch_bar.set_postfix(
                            train_loss=f"{current_train_loss:.4f}",
                            val_loss=f"{current_eval_loss:.4f}",
                        )
        return train_history, eval_history


##### Pure version of methods for jit compilation #####

# ── Standard VAE warmup (closed-form KL, no GMM) ────────────────────────────

@nnx.jit
def _compute_loss_vae_warmup(model: GMMVAE, batch: jnp.array, key: jax.Array) -> jnp.array:
    """Standard VAE ELBO with N(0,I) prior and closed-form KL.
    """
    def elbo_single(x, key):
        mu, _ = model.encoder(x)  # ignore logvar entirely
        p_dec = jnp.clip(model.decoder(mu), 1e-7, 1.0 - 1e-7)
        recon = jnp.sum(x * jnp.log(p_dec) + (1 - x) * jnp.log(1 - p_dec))
        return recon

    keys = jr.split(key, batch.shape[0])
    return -jnp.mean(jax.vmap(elbo_single)(batch, keys))


@nnx.jit
def _train_step_vae_warmup(model: GMMVAE, optimizer: nnx.Optimizer,
                            metrics: nnx.MultiMetric, batch: jnp.array):
    val_and_grad_fn = nnx.value_and_grad(_compute_loss_vae_warmup, argnums=0)
    val, grads = val_and_grad_fn(model, batch, model.rngs.param())
    optimizer.update(model, grads)
    metrics.update(loss=val)


@nnx.jit
def _eval_step_vae_warmup(model: GMMVAE, metrics: nnx.MultiMetric, batch: jnp.array):
    loss = _compute_loss_vae_warmup(model, batch, model.rngs.param())
    metrics.update(loss=loss)


# ── GMM-VAE fine-tuning ──────────────────────────────────────────────────────

@nnx.jit
def _gmm_logpdf(z, logit_alpha, mu, logvar):
    def gaussian_logpdf_single(mu_single, logvar_single):
        cov = jnp.diag(jnp.exp(logvar_single))
        return multivariate_normal.logpdf(z, mu_single, cov)
    log_gauss_comp = jax.vmap(gaussian_logpdf_single)(mu, logvar)
    log_weights = jax.nn.log_softmax(logit_alpha)
    return jax.scipy.special.logsumexp(log_weights + log_gauss_comp)


@nnx.jit
def _compute_loss_gmm(model : GMMVAE, batch : jnp.array, key : jax.Array)->jnp.array:
    #Get model's parameters
    logit_alpha_gmm = model.logit_alpha_gmm.get_value()
    mu_gmm = model.mu_gmm.get_value()
    logvar_gmm = model.logvar_gmm.get_value()
    beta = model.beta

    def elbo_single(x : jnp.array, key : jax.Array)->jnp.array:
        # Computes mean and variance of the encoder
        latent_dim = model.latent_dim
        mu_enc, logvar_enc = model.encoder(x)

        # Sample from q_\phi(z|x) with reparametrization trick
        z = mu_enc + jnp.exp(0.5*logvar_enc)*jr.normal(key, shape = (latent_dim,))

        # Computes p_\theta(z)
        p_dec = model.decoder(z)
        p_dec = jnp.clip(p_dec, 1e-7, 1.0 - 1e-7)

        # Computes the ELBO
        term_prior = _gmm_logpdf(z, logit_alpha_gmm, mu_gmm, logvar_gmm)
        term_dec = jnp.sum(x*jnp.log(p_dec) + (1-x)*jnp.log(1-p_dec))
        term_enc = multivariate_normal.logpdf(z, mu_enc, jnp.diag(jnp.exp(logvar_enc)))
        elbo = term_dec + beta*(term_prior - term_enc)
        return elbo

    keys = jr.split(key, batch.shape[0])
    elbo = jnp.mean(jax.vmap(elbo_single)(batch, keys))
    return -elbo


@nnx.jit
def _train_step_gmm(model : GMMVAE, optimizer : nnx.Optimizer, metrics : nnx.MultiMetric, batch : jnp.array):
    val_and_grad_fn = nnx.value_and_grad(_compute_loss_gmm, argnums = 0)
    val, grads = val_and_grad_fn(model, batch, model.rngs.param())
    optimizer.update(model, grads)
    metrics.update(loss = val)

@nnx.jit
def _eval_step_gmm(model : GMMVAE, metrics : nnx.MultiMetric, batch : jnp.ndarray):
    loss = _compute_loss_gmm(model, batch, model.rngs.param())
    metrics.update(loss = loss)
