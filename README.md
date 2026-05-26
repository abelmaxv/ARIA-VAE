# ARIA-VAE

Une implémentation JAX/Flax de plusieurs variantes d'auto-encodeurs variationnels (VAE), évaluée sur MNIST binarisé. Le projet est structuré comme une base de code de recherche accompagnant un rapport écrit, avec un module par famille de modèles et un notebook Jupyter par expérience.

---

## Modèles

| Modèle | Classe | Prior | Notes |
|--------|--------|-------|-------|
| **VAE classique** | `ClassicVAE` | Gaussien $\mathcal{N}(0,I)$ | Encodeur/décodeur linéaire ou convolutionnel |
| **GMM-VAE** | `GMMVAE` | Mélange de Gaussiennes (K composantes) | Prior appris ou fixe ; supporte la pondération $\beta$-VAE |
| **VaDE** | `VaDE` | Mélange de Gaussiennes (K composantes) | Ajoute un encodeur catégoriel (LeNet-5) pour $q(c\|x)$ |

### VAE classique
$\beta$-VAE standard avec décodeur de Bernoulli. Deux architectures encodeur/décodeur sont disponibles :
- **`arch="linear"`** — perceptron multicouche à une couche cachée
- **`arch="conv"`** — encodeur convolutionnel à trois couches avec stride, décodeur linéaire symétrique

### GMM-VAE
Remplace le prior gaussien isotrope par un mélange de K Gaussiennes $p(z) = \sum_k \alpha_k \mathcal{N}(z; \mu_k, \text{diag}(\sigma_k^2))$. Les poids $\alpha_k$, moyennes $\mu_k$ et log-variances sont soit **appris de bout en bout** (`learn_prior=True`) soit **fixes** (`learn_prior=False`). Un coefficient $\beta$ contrôle le poids du terme KL.

### VaDE (Variational Deep Embedding)
Augmente le GMM-VAE avec un posterior catégoriel explicite $q(c|x)$ implémenté sous forme de classifieur LeNet-5. L'ELBO se décompose en un terme de reconstruction, un terme KL gaussien par composante pondéré par $q(c|x)$, et un terme d'entropie sur la variable discrète.

---

## Structure du projet

```
ARIA-VAE/
├── model/
│   ├── classicVAE.py       # Blocs Encodeur/Décodeur + ClassicVAE
│   ├── gmmVAE.py           # GMMVAE (prior GMM, β optionnel, prior appris optionnel)
│   └── vade.py             # VaDE (prior GMM + encodeur catégoriel)
├── train/
│   ├── classicVAE_train.py # ClassicVAE_trainer
│   ├── gmmVAE_train.py     # GMMVAE_trainer (pretrain + init_gmm_kmeans + train)
│   └── vade_train.py       # VaDE_trainer   (pretrain + init_gmm_kmeans + train)
├── test/
│   ├── classicVAE_test.ipynb       # Expériences VAE linéaire et convolutionnel
│   ├── gmmVAE_test.ipynb           # GMM-VAE (prior appris)
│   ├── gmmVAE_fixed_test.ipynb     # GMM-VAE (prior fixe)
│   ├── beta_gmmVAE_test.ipynb      # GMM β-VAE (β = 50)
│   ├── vade_test.ipynb             # VaDE
│   ├── vade_fixed_test.ipynb       # VaDE (prior fixe)
│   └── comparative_test.ipynb      # VAE conv, GMM-VAE, VaDE — reconstruction, génération, espace latent
├── checkpoints/            # Poids sauvegardés (orbax) après entraînement dans comparative_test
├── report/
│   ├── report.tex
│   └── img/               # Figures pour le rapport et l'annexe
├── utils.py                # image_to_jax1d, jax_collate_fn, plot_image
└── data/                   # MNIST téléchargé automatiquement ici
```

---

## Installation

> Testé avec Python 3.12 sur macOS (Apple Silicon, backend CPU).

```bash
conda create -n aria_fondements python=3.12
conda activate aria_fondements

# JAX (CPU)
pip install jax

# Pile d'apprentissage profond
pip install flax optax

# Données — PyTorch est utilisé uniquement pour le DataLoader et MNIST de torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Visualisation et analyse
pip install matplotlib seaborn pandas scipy pcax umap-learn tqdm

# Sauvegarde des poids
pip install orbax-checkpoint
```

> **Note Apple Silicon :** définir `os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"` et `os.environ["JAX_PLATFORMS"] = "cpu"` avant d'importer JAX (déjà fait dans tous les notebooks).

---

## Démarrage rapide

### VAE classique

```python
from model import ClassicVAE
from train import ClassicVAE_trainer
from utils import image_to_jax1d, jax_collate_fn

from flax import nnx
import optax
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

train_loader = DataLoader(
    MNIST(root="./data", train=True, download=True, transform=image_to_jax1d),
    batch_size=128, shuffle=True, collate_fn=jax_collate_fn,
)
test_loader = DataLoader(
    MNIST(root="./data", train=False, download=True, transform=image_to_jax1d),
    batch_size=128, shuffle=True, collate_fn=jax_collate_fn,
)

model     = ClassicVAE(in_dim=784, latent_dim=16, hidden_dim=256, rngs=nnx.Rngs(0), arch="conv")
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
trainer   = ClassicVAE_trainer(model, optimizer)

train_history, eval_history = trainer.train(epochs=100, train_data_loader=train_loader, test_dataloader=test_loader)
```

### GMM-VAE et VaDE — entraînement en trois phases

Les modèles à prior mélange de gaussiennes suivent un protocole en trois étapes :

```python
from model import GMMVAE, VaDE
from train import GMMVAE_trainer, VaDE_trainer

model     = GMMVAE(in_dim=784, latent_dim=16, hidden_dim=256, K=10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
trainer   = GMMVAE_trainer(model, optimizer)

# Phase 1 — pré-entraînement VAE avec prior N(0, I)
trainer.pretrain(epochs=10, train_data_loader=train_loader, test_dataloader=test_loader)

# Phase 2 — initialisation des paramètres GMM par K-means sur les encodages
trainer.init_gmm_kmeans(train_data_loader=train_loader)

# Phase 3 — fine-tuning avec l'ELBO GMM complet
train_history, eval_history = trainer.train(epochs=100, train_data_loader=train_loader, test_dataloader=test_loader)
```

La même interface s'applique à `VaDE_trainer`.

### Échantillonnage et reconstruction

```python
import jax.random as jr

key = jr.PRNGKey(0)

# Générer un nouvel échantillon depuis le prior
sample = model.generate_mean(key)        # forme (784,), moyenne du décodeur (sans bruit)

# Encoder une image vers un vecteur latent
z = model.encode(image, key)             # forme (latent_dim,)

# Reconstruire (encoder puis décoder)
reconstruction = model(image, key)       # forme (784,)
```

### Sauvegarde et chargement des poids

```python
import orbax.checkpoint as ocp
from pathlib import Path
from flax import nnx

checkpointer = ocp.StandardCheckpointer()
ckpt_dir     = Path("./checkpoints").resolve()

# Sauvegarde
state = nnx.state(model, nnx.Param)
checkpointer.save(ckpt_dir / "model_gmm", state, force=True)
checkpointer.wait_until_finished()

# Chargement (le modèle doit être instancié avec la même architecture)
state    = nnx.state(model, nnx.Param)
restored = checkpointer.restore(ckpt_dir / "model_gmm", state)
nnx.update(model, restored)
```

> **Note :** orbax requiert des chemins absolus — utiliser `.resolve()` sur tout objet `Path`.

---

## Expériences

### Notebooks individuels

Chaque notebook dans `test/` suit la même structure :

1. **Entraînement** — instanciation du modèle, configuration de l'optimiseur, boucle d'entraînement
2. **Reconstruction** — comparaison côte à côte des images originales et reconstruites
3. **Génération** — échantillons tirés depuis le prior
4. **Espace latent** — projection PCA et UMAP des encodages colorés par classe, avec contour de densité du prior

### Notebook comparatif

`comparative_test.ipynb` compare les trois modèles retenus — **VAE convolutionnel**, **GMM-VAE (prior appris)** et **VaDE** — en trois figures unifiées :

| Figure | Disposition |
|--------|-------------|
| Reconstruction | Grille 4 × 5 (Original + 3 modèles) |
| Génération | Grille 3 × 5 |
| Espace latent (PCA) | 3 sous-figures côte à côte avec contour de densité du prior et moyennes GMM |
| Espace latent (UMAP) | 3 sous-figures côte à côte avec centres de classes/composantes |

Les poids des trois modèles sont sauvegardés dans `checkpoints/` après entraînement. Une cellule de chargement (après la cellule d'entraînement du VAE conv) permet de relancer le notebook sans réentraîner.

---

## Dépendances

| Paquet | Rôle |
|--------|------|
| `jax` / `jaxlib` | Opérations sur les tableaux, compilation JIT, différentiation automatique |
| `flax` (API nnx) | Modules de réseaux de neurones, wrapper d'optimiseur |
| `optax` | Optimiseur Adam |
| `orbax-checkpoint` | Sauvegarde et chargement des poids de modèles |
| `torch` / `torchvision` | Téléchargement de MNIST et `DataLoader` uniquement |
| `pcax` | ACP sur des tableaux JAX |
| `umap-learn` | Réduction de dimension UMAP |
| `matplotlib` / `seaborn` | Visualisation |
| `pandas` / `numpy` / `scipy` | Manipulation de données et statistiques |
| `tqdm` | Barres de progression lors de l'entraînement |
