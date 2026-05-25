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
│   ├── gmmVAE_train.py     # GMMVAE_trainer
│   └── vade_train.py       # VaDE_trainer
├── test/
│   ├── classicVAE_test.ipynb       # Expériences VAE linéaire et convolutionnel
│   ├── gmmVAE_test.ipynb           # GMM-VAE (prior appris)
│   ├── gmmVAE_fixed_test.ipynb     # GMM-VAE (prior fixe)
│   ├── beta_gmmVAE_test.ipynb      # GMM β-VAE (β = 50)
│   ├── vade_test.ipynb             # VaDE (prior appris)
│   ├── vade_fixed_test.ipynb       # VaDE (prior fixe)
│   └── comparative_test.ipynb      # Tous les modèles — reconstruction, génération, espace latent
├── report/
│   └── report.tex
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
```

> **Note Apple Silicon :** définir `os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"` et `os.environ["JAX_PLATFORMS"] = "cpu"` avant d'importer JAX (déjà fait dans tous les notebooks).

---

## Démarrage rapide

### Entraînement d'un modèle

```python
from model import ClassicVAE, GMMVAE, VaDE
from train import ClassicVAE_trainer, GMMVAE_trainer, VaDE_trainer
from utils import image_to_jax1d, jax_collate_fn

from flax import nnx
import optax
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Données
train_dataset = MNIST(root="./data", train=True, download=True, transform=image_to_jax1d)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=jax_collate_fn)
test_loader   = DataLoader(
    MNIST(root="./data", train=False, download=True, transform=image_to_jax1d),
    batch_size=128, shuffle=True, collate_fn=jax_collate_fn,
)

# VAE classique (convolutionnel)
model     = ClassicVAE(in_dim=784, latent_dim=16, hidden_dim=256, rngs=nnx.Rngs(0), arch="conv")
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
trainer   = ClassicVAE_trainer(model, optimizer)

train_history, eval_history = trainer.train(epochs=10, train_data_loader=train_loader, test_dataloader=test_loader)
```

Les trois classes d'entraînement partagent la même interface :

```python
train_history, eval_history = trainer.train(epochs, train_data_loader, test_dataloader)
```

### Échantillonnage et reconstruction

```python
import jax, jax.random as jr

key = jr.PRNGKey(0)

# Générer un nouvel échantillon depuis le prior
sample = model.generate(key)                        # forme (784,)

# Encoder une image vers un vecteur latent
z = model.encode(image, key)                        # forme (latent_dim,)

# Reconstruire (encoder puis décoder, retourne un échantillon de Bernoulli)
reconstruction = model(image, key)                  # forme (784,)
```

### Variantes GMM-VAE

```python
# Prior GMM appris (par défaut)
model = GMMVAE(in_dim=784, latent_dim=16, hidden_dim=256, K=10, rngs=nnx.Rngs(0))

# Prior GMM fixe
model = GMMVAE(in_dim=784, latent_dim=16, hidden_dim=256, K=10, rngs=nnx.Rngs(0), learn_prior=False)

# β-VAE avec prior GMM
model = GMMVAE(in_dim=784, latent_dim=16, hidden_dim=256, K=10, rngs=nnx.Rngs(0), beta=50)

# Accéder aux paramètres GMM appris
alphas = jax.nn.softmax(model.logit_alpha_gmm.get_value())   # poids du mélange
mu     = model.mu_gmm.get_value()                            # moyennes des composantes (K, latent_dim)
```

---

## Expériences

Chaque notebook dans `test/` suit la même structure :

1. **Entraînement** — instanciation du modèle, configuration de l'optimiseur, boucle d'entraînement
2. **Reconstruction** — comparaison côte à côte des images originales et reconstruites
3. **Génération** — échantillons tirés depuis le prior
4. **Espace latent** — projection PCA des données d'entraînement encodées, colorées par classe de chiffre, avec contour de densité du prior

`comparative_test.ipynb` combine les sept variantes de modèles en trois figures unifiées :

| Figure | Disposition |
|--------|-------------|
| Reconstruction | Grille 8 × 5 (Original + 7 modèles) |
| Génération | Grille 7 × 5 |
| Espace latent (ACP) | Grille de sous-figures 4 × 2 avec contour de densité du prior + moyennes GMM |
| Espace latent (UMAP) | Grille de sous-figures 4 × 2 avec centres de classes/composantes |

---

## Dépendances

| Paquet | Rôle |
|--------|------|
| `jax` / `jaxlib` | Opérations sur les tableaux, compilation JIT, différentiation automatique |
| `flax` (API nnx) | Modules de réseaux de neurones, wrapper d'optimiseur |
| `optax` | Optimiseur Adam |
| `torch` / `torchvision` | Téléchargement de MNIST et `DataLoader` uniquement |
| `pcax` | ACP sur des tableaux JAX |
| `umap-learn` | Réduction de dimension UMAP |
| `matplotlib` / `seaborn` | Visualisation |
| `pandas` / `numpy` / `scipy` | Manipulation de données et statistiques |
| `tqdm` | Barres de progression lors de l'entraînement |
