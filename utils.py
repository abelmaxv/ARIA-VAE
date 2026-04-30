import PIL
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def image_to_jax1d(image: PIL.Image.Image) -> jnp.array:
    """Converts a PIL image to a flat binary JAX array with values in {0.0, 1.0}.

    Args:
        image (PIL.Image.Image): Grayscale PIL image to convert.

    Returns:
        jnp.array: 1D float32 array of length width * height with binarized pixel values.
    """
    # Convert to numpy first to avoid Metal backend memory placement issues
    image_np = np.array(image.getdata()).reshape(image.width * image.height)
    image_jax = jnp.array(image_np)
    image_jax = jnp.where(image_jax > 128, 1.0, 0.0).astype(jnp.float32)
    return image_jax

def jax_collate_fn(batch)->tuple[jnp.array, jnp.array]:
    """Collates a list of (image, label) samples into batched JAX arrays.

    Args:
        batch (list[tuple]): List of (image, label) pairs returned by a Dataset.

    Returns:
        tuple[jnp.array, jnp.array]: Batched images of shape (N, ...) and corresponding labels.
    """
    images, labels = zip(*batch)
    images_batch = jnp.stack(images, axis=0).astype(jnp.float32)
    labels_batch = jnp.array(labels)
    return images_batch, labels_batch


def plot_image(image: jnp.array, ax=None):
    """Displays a flat JAX array as a 28x28 grayscale image, optionally on a provided axis.

    Args:
        image (jnp.array): 1D array of length 784 representing a grayscale image.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, uses plt.imshow.
    """
    image = image.reshape(28, 28)
    if ax is not None:
        ax.imshow(image, cmap="gray")
    else:
        plt.imshow(image, cmap="gray")
