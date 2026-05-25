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
        """Trains the model for a given number of epochs and returns loss histories.

        Args:
            epochs (int): Number of training epochs.
            train_data_loader (DataLoader): DataLoader for the training set.
            test_dataloader (DataLoader): DataLoader for the evaluation set.
            eval_every (int): Number of batches between evaluation steps.

        Returns:
            tuple[list, list]: Train loss history and eval loss history.
        """
        train_history = []
        eval_history = []

        # Loading bar for epochs: 
        epoch_bar = tqdm(range(epochs), desc='Training')
        with jax.debug_nans(True):
            for epoch in epoch_bar:
                
                # Loading bar for batches
                batch_bar = tqdm(train_data_loader, desc=f"Epoch {epoch}", leave=False)

                for batch_id, batch_data in enumerate(batch_bar):

                    # Train step : 
                    self.model.train() # Pas vraiment utile car pas de dropout ...
                    self.train_step(self.model, self.optimizer, self.metrics, batch_data[0])

                    # Eval step :
                    if batch_id>0 and (batch_id%eval_every == 0 or batch_id == len(train_data_loader)-1):
                        train_metrics = self.metrics.compute()
                        current_train_loss = float(train_metrics["loss"])
                        train_history.append(current_train_loss)
                        self.metrics.reset()
                        
                        self.model.eval()
                        for _, test_batch_data in enumerate(test_dataloader):
                            self.eval_step(self.model, self.metrics, test_batch_data[0])

                        eval_metrics = self.metrics.compute()
                        current_eval_loss = float(eval_metrics["loss"])
                        eval_history.append(current_eval_loss)
                        self.metrics.reset()

                        batch_bar.set_postfix(train_loss=f"{current_train_loss:.4f}", 
                                           val_loss=f"{current_eval_loss:.4f}")
        return train_history, eval_history


##### Pure version of methods for jit compilation #####
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
def _compute_loss_vade(model : VaDE, batch : jnp.array, key : jax.Array)->jnp.array:
    #Get model's parameters
    logit_alpha_gmm = model.logit_alpha_gmm.get_value()
    mu_gmm = model.mu_gmm.get_value()
    logvar_gmm = model.logvar_gmm.get_value()
    beta = model.beta

    def elbo_single(x : jnp.array, key : jax.Array)->jnp.array: 
        # Computes mean and variance of the encoder
        latent_dim = model.latent_dim
        mu_enc, logvar_enc = model.encoder(x)

        # Computes the class probabilities by surrogate posterior
        class_prob = model.class_prob(x)

        # Sample from q_\phi(z|x) with reparametrization trick
        z = mu_enc + jnp.exp(0.5*logvar_enc)*jr.normal(key, shape = (latent_dim,))

        # Computes p_\theta(z)
        p_dec = model.decoder(z)
        p_dec = jnp.clip(p_dec, 1e-7, 1.0 - 1e-7)

        # Computes the ELBO
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
