import sys
from pathlib import Path

# # Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.classicVAE import ClassicVAE
from utils import image_to_jax1d, jax_collate_fn

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import PIL

from tqdm import tqdm


# TO DO : 
# - Implement nnx.metric
# - Accelerate the training with jax.jit

class ClassicVAE_trainer: 
    """_summary_
    """

    def __init__(self, model: ClassicVAE, optimizer: nnx.Optimizer, train_data_loader: DataLoader, test_data_loader : DataLoader):
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader



    def loss_function(self, model: ClassicVAE, image_batch : tuple[jnp.array, jnp.array], n_sample : int, key : jnp.array) -> jnp.array:
        """ TO DO
        """

        latent_dim = model.latent_dim
        

        def loss_single(x: jnp.array, n_sample : int) -> jnp.array:
            mu_x, log_sigma_x = model.encoder(x)
            zs = mu_x[None, :] + jnp.exp(log_sigma_x)[None, :] * jr.normal(key, shape=(n_sample, latent_dim))
            
            def log_likelihood(z: jnp.array) -> jnp.array:
                ps = model.decoder(z)
                return jnp.sum(jnp.log(ps)*x + jnp.log(1-ps)*(1-x))

            decoder_term = -jnp.mean(nnx.vmap(log_likelihood)(zs))
            encoder_term = jnp.sum(0.5*(mu_x**2 + jnp.exp(log_sigma_x)**2-1)-log_sigma_x)
            return decoder_term + encoder_term
        
        return jnp.mean(nnx.vmap(lambda image : loss_single(image, n_sample))(image_batch))

            

    def _train_step(self, batch : tuple[jnp.array, jnp.array], *, key : jnp.array, n_sample : int):
        image_batch, labels_batch = batch
        grad_fn = nnx.grad(self.loss_function, argnums = 0)
        grads = grad_fn(self.model, image_batch, n_sample, key = key)
        self.optimizer.update(grads)


    def _eval_step(self, batch : tuple[jnp.array, jnp.array])-> jnp.array:
        # To be implemented with metrics
        pass

    def __call__(self, epochs: int, n_sample : int = 10, *, rngs: nnx.Rngs):
        """ Launch the training procedure
        """
        for epoch in range(epochs): 
            key = rngs.params()
            for batch in tqdm(self.train_data_loader): 
                key, subkey = jr.split(key)
                self._train_step(batch, key = subkey, n_sample = n_sample)
                # To be implemented
                # if eval: 
                #     self._eval_step(batch)
        


if __name__ == "__main__":

    # Test if the methods run without errors

    train_dataset = MNIST(root = "./../data", train = True, download = True, transform = image_to_jax1d)
    test_dataset = MNIST(root = "./../data", train = False, download = True, transform = image_to_jax1d)

    train_data_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, collate_fn = jax_collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size = 128, shuffle = True, collate_fn = jax_collate_fn)

    in_dim = train_dataset[0][0].shape[0]
    latent_dim = 64
    hidden_dim = 128
    rngs = nnx.Rngs(jax.random.PRNGKey(42))
    learning_rate = 0.001
    epochs = 1

    model = ClassicVAE(in_dim = in_dim, latent_dim = latent_dim, hidden_dim = hidden_dim, rngs = rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    trainer = ClassicVAE_trainer(model, optimizer, train_data_loader=train_data_loader, test_data_loader=test_data_loader)

    trainer(epochs = epochs, rngs = rngs)