import PIL 
import jax.numpy as jnp
import matplotlib.pyplot as plt


def image_to_jax1d(image: PIL.Image.Image) -> jnp.array:
    image_jax = jnp.array(image.getdata()).reshape(image.width * image.height)
    image_jax = jnp.where(image_jax > 128, 1.0, 0.0).astype(jnp.float32)
    return image_jax

def jax_collate_fn(batch):
    """Custom collate function for JAX arrays"""
    images, labels = zip(*batch)
    # Stack JAX arrays into batched arrays and ensure float dtype
    images_batch = jnp.stack(images, axis=0).astype(jnp.float32)
    labels_batch = jnp.array(labels)
    return images_batch, labels_batch


def plot_image(image: jnp.array):
    image = image.reshape(28, 28)
    plt.imshow(image, cmap = "gray")