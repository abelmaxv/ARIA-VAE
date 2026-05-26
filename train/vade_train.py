import sys
from pathlib import Path

# # Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.vade import VaDE

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2




class VaDE_trainer:

    def __init__(self, model: VaDE, optimizer: nnx.Optimizer):
        """
        Args:
            model (VaDE): The VAE model to train.
            optimizer (nnx.Optimizer): Optimizer wrapping the model parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.metrics = nnx.MultiMetric(loss = nnx.metrics.Average(argname='loss'))


    def compute_loss(self, model : VaDE, batch : jnp.array, key : jax.Array)->jnp.array:
        """Computes the negative ELBO loss for a batch.

        Args:
            model (VaDE): The VAE model.
            batch (jnp.array): Batch of input images of shape (N, in_dim).
            key (jax.Array): PRNG key for sampling.

        Returns:
            jnp.array: Scalar loss value (negative ELBO).
        """
        return _compute_loss_vade(model, batch, key)



    def train_step(self, model : VaDE, optimizer : nnx.Optimizer, metrics : nnx.MultiMetric, batch : jnp.array):
        """Runs one JIT-compiled training step: computes gradients, updates parameters, and logs the loss.

        Args:
            model (VaDE): The VAE model.
            optimizer (nnx.Optimizer): Optimizer to apply gradients.
            metrics (nnx.MultiMetric): Metrics accumulator.
            batch (jnp.array): Batch of input images of shape (N, in_dim).
        """
        _train_step_vade(model, optimizer, metrics, batch)


    def eval_step(self, model : VaDE, metrics : nnx.MultiMetric, batch : jnp.ndarray) -> jnp.ndarray:
        """Runs one JIT-compiled evaluation step: computes the loss and logs it without updating parameters.

        Args:
            model (VaDE): The VAE model.
            metrics (nnx.MultiMetric): Metrics accumulator.
            batch (jnp.ndarray): Batch of input images of shape (N, in_dim).
        """
        _eval_step_vade(model, metrics, batch)


    def train(self, epochs : int , train_data_loader : DataLoader, test_dataloader : DataLoader, eval_every : int = 50) -> tuple[list, list]:
        """Fine-tunes the model with the full VaDE ELBO.

        Args:
            epochs (int): Number of training epochs.
            train_data_loader (DataLoader): DataLoader for the training set.
            test_dataloader (DataLoader): DataLoader for the evaluation set.
            eval_every (int): Number of batches between evaluation steps.

        Returns:
            tuple[list, list]: Train loss history and eval loss history.
        """
        return self._run_loop(epochs, train_data_loader, test_dataloader,
                              eval_every, desc='Fine-tuning (VaDE)',
                              train_fn=_train_step_vade,
                              eval_fn=_eval_step_vade)

    # ------------------------------------------------------------------ #
    #  Phase 1 — VAE pre-training with Gaussian prior N(0, I)             #
    # ------------------------------------------------------------------ #

    def pretrain(self, epochs: int, train_data_loader: DataLoader,
                 test_dataloader: DataLoader, eval_every: int = 50) -> tuple[list, list]:
        """Phase 1: pre-train encoder/decoder with a standard Gaussian VAE ELBO.

        The GMM parameters are not used and receive zero gradients during this
        phase.  The closed-form KL divergence KL(q(z|x) ‖ N(0,I)) is used so
        no Monte-Carlo estimation is needed for the KL term.

        Call this before :meth:`init_gmm_kmeans` and :meth:`train`.

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

    # ------------------------------------------------------------------ #
    #  Phase 2 — K-means initialisation of GMM parameters                #
    # ------------------------------------------------------------------ #

    def init_gmm_kmeans(self, train_data_loader: DataLoader, seed: int = 42) -> np.ndarray:
        """Phase 2: initialise GMM parameters with K-means on the encoded training set.

        After the encoder has been pre-trained, this method:
          1. Encodes every training image to obtain its posterior mean μ_φ(x).
          2. Runs K-means (K = model.K) on those means.
          3. Sets the GMM parameters of the model:
               • μ_k      ← centroid of cluster k
               • logvar_k ← log(per-dimension variance of points in cluster k)
               • α_k      ← fraction of training points assigned to cluster k

        Works for both ``learn_prior=True`` (nnx.Param) and ``learn_prior=False``
        (nnx.Variable) — the values are assigned directly regardless of type.

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
        logvar_init = np.zeros((K, D))
        counts      = np.zeros(K)

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

    # ------------------------------------------------------------------ #
    #  Shared training loop                                               #
    # ------------------------------------------------------------------ #

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
def _compute_loss_vae_warmup(model: VaDE, batch: jnp.array, key: jax.Array) -> jnp.array:
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
def _train_step_vae_warmup(model: VaDE, optimizer: nnx.Optimizer,
                            metrics: nnx.MultiMetric, batch: jnp.array):
    val_and_grad_fn = nnx.value_and_grad(_compute_loss_vae_warmup, argnums=0)
    val, grads = val_and_grad_fn(model, batch, model.rngs.param())
    optimizer.update(model, grads)
    metrics.update(loss=val)


@nnx.jit
def _eval_step_vae_warmup(model: VaDE, metrics: nnx.MultiMetric, batch: jnp.array):
    loss = _compute_loss_vae_warmup(model, batch, model.rngs.param())
    metrics.update(loss=loss)


# ── VaDE fine-tuning ─────────────────────────────────────────────────────────

@nnx.jit
def _vade_log_prior(z, logit_alpha, mu, logvar, class_prob):
    def gaussian_logpdf_single(mu_single, logvar_single):
        cov = jnp.diag(jnp.exp(logvar_single))
        return multivariate_normal.logpdf(z, mu_single, cov)
    log_gauss_comp = jax.vmap(gaussian_logpdf_single)(mu, logvar)
    log_weights = jax.nn.log_softmax(logit_alpha)
    return (class_prob*log_weights).sum() + (class_prob*log_gauss_comp).sum()

@nnx.jit
def _vade_log_posterior(z, mu_enc, logvar_enc, class_prob):
    z_term = multivariate_normal.logpdf(z, mu_enc, jnp.diag(jnp.exp(logvar_enc)))
    safe_class_prob = jnp.where(class_prob > 0, class_prob, jnp.ones_like(class_prob))
    class_term = (class_prob*jnp.log(safe_class_prob)).sum()
    return z_term + class_term

@nnx.jit
def _vade_class_prob_from_z(z, logit_alpha, mu_gmm, logvar_gmm):
    """Analytical q(c | x) = p(c | z) evaluated at the sampled z (Eq. 16 of VaDE paper).

    q(c_j | x) = p(c_j) · p(z | c_j) / Σ_k p(c_k) · p(z | c_k)

    Using z (not μ_enc) ensures D_KL(q(c|x) ‖ p(c|z)) = 0 at the optimum,
    as derived in the paper.
    """
    def log_component(mu_j, logvar_j):
        cov_j = jnp.diag(jnp.exp(logvar_j))
        return multivariate_normal.logpdf(z, mu_j, cov_j)   # evaluated at z

    log_gauss = jax.vmap(log_component)(mu_gmm, logvar_gmm)  # (K,)
    log_alpha = jax.nn.log_softmax(logit_alpha)               # (K,)
    return jnp.exp(jax.nn.log_softmax(log_alpha + log_gauss)) # (K,)


@nnx.jit
def _compute_loss_vade(model : VaDE, batch : jnp.array, key : jax.Array)->jnp.array:
    #Get model's parameters
    logit_alpha_gmm = model.logit_alpha_gmm.get_value()
    mu_gmm = model.mu_gmm.get_value()
    logvar_gmm = model.logvar_gmm.get_value()
    beta = model.beta

    def elbo_single(x : jnp.array, key : jax.Array)->jnp.array:
        latent_dim = model.latent_dim
        mu_enc, logvar_enc = model.encoder(x)

        # Sample z first — class_prob is computed from z per Eq. 16
        z = mu_enc + jnp.exp(0.5*logvar_enc)*jr.normal(key, shape=(latent_dim,))

        # q(c | x) = p(c | z)  evaluated at the sampled z
        class_prob = _vade_class_prob_from_z(z, logit_alpha_gmm, mu_gmm, logvar_gmm)

        # Decoder output
        p_dec = model.decoder(z)
        p_dec = jnp.clip(p_dec, 1e-7, 1.0 - 1e-7)

        # ELBO terms
        term_prior = _vade_log_prior(z, logit_alpha_gmm, mu_gmm, logvar_gmm, class_prob)
        term_dec = jnp.sum(x*jnp.log(p_dec) + (1-x)*jnp.log(1-p_dec))
        term_enc = _vade_log_posterior(z, mu_enc, logvar_enc, class_prob)
        elbo = term_dec + beta*(term_prior - term_enc)
        return elbo

    keys = jr.split(key, batch.shape[0])
    elbo = jnp.mean(jax.vmap(elbo_single)(batch, keys))
    return -elbo


@nnx.jit
def _train_step_vade(model : VaDE, optimizer : nnx.Optimizer, metrics : nnx.MultiMetric, batch : jnp.array):
    val_and_grad_fn = nnx.value_and_grad(_compute_loss_vade, argnums = 0)
    val, grads = val_and_grad_fn(model, batch, model.rngs.param())
    optimizer.update(model, grads)
    metrics.update(loss = val)

@nnx.jit
def _eval_step_vade(model : VaDE, metrics : nnx.MultiMetric, batch : jnp.ndarray):
    loss = _compute_loss_vade(model, batch, model.rngs.param())
    metrics.update(loss = loss)
