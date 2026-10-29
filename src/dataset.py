"""
dataset.py — Pregunta 1
Carga de CIFAR-10, semilla individual, visualización, pipeline de
transformaciones (con Data Augmentation) y partición estratificada.

Estudiante : Jerson Americo Peña Salgueron
Código     : 021101457b  ->  semilla = 57 (dos últimos dígitos numéricos)

Ejecución (desde la raíz del repositorio):
    python src/dataset.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 1. Configuración general y semilla individual
# ---------------------------------------------------------------------------
SEED = 57  # Código de matrícula 021101457b -> dos últimos dígitos: 57

ROOT_DIR = Path(__file__).resolve().parent.parent   # raíz del repositorio
DATA_DIR = ROOT_DIR / "data"                       # aquí se descarga CIFAR-10 (no se sube a GitHub)
OUTPUT_DIR = ROOT_DIR / "outputs"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Media y desviación estándar por canal (R, G, B) del conjunto de
# entrenamiento de CIFAR-10. Son los valores de referencia usados
# habitualmente para este dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad para que la ejecución sea reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Hace que cuDNN elija algoritmos deterministas (en GPU).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Semilla para cada proceso 'worker' del DataLoader (si num_workers > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# 2. Pipeline de transformaciones
# ---------------------------------------------------------------------------
def get_transforms():
    """
    Devuelve (transformación de entrenamiento, transformación de evaluación).

    - Entrenamiento: Data Augmentation aleatorio + ToTensor + Normalize.
    - Validación / prueba: solo ToTensor + Normalize (determinista), para que
      la medición del rendimiento no dependa del azar.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),        # recorte aleatorio con relleno de 4 px
        transforms.RandomHorizontalFlip(p=0.5),      # espejo horizontal con prob. 0.5
        transforms.ToTensor(),                       # PIL [0,255] HWC -> tensor [0,1] CHW
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # (x - media) / desv. por canal
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Revierte la normalización para poder visualizar una imagen (C, H, W)."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# 3. Carga del dataset
# ---------------------------------------------------------------------------
def load_raw_datasets(data_dir: Path = DATA_DIR):
    """Descarga (solo la primera vez) y carga CIFAR-10 sin transformaciones."""
    train_raw = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_raw = datasets.CIFAR10(root=data_dir, train=False, download=True)
    return train_raw, test_raw


def class_distribution(targets, indices=None) -> dict:
    """Cuenta cuántas imágenes hay por clase (opcionalmente en un subconjunto de índices)."""
    targets = np.asarray(targets)
    if indices is not None:
        targets = targets[indices]
    counts = np.bincount(targets, minlength=len(CLASSES))
    return {CLASSES[i]: int(c) for i, c in enumerate(counts)}


# ---------------------------------------------------------------------------
# 4. Partición estratificada
# ---------------------------------------------------------------------------
def stratified_split(targets, val_size: float = 0.2, seed: int = SEED):
    """
    Divide los índices del conjunto oficial de entrenamiento (50 000 imágenes)
    en entrenamiento y validación, conservando la proporción de cada clase.
    """
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,      # misma proporción de clases en ambas particiones
        random_state=seed,     # semilla 57 -> partición reproducible
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx)


def get_dataloaders(batch_size: int = 64, val_size: float = 0.2,
                    num_workers: int = 0, seed: int = SEED, verbose: bool = True):
    """
    Construye los DataLoaders de entrenamiento, validación y prueba.

    Se crean DOS instancias del conjunto oficial de entrenamiento: una con
    Data Augmentation (para el subconjunto de entrenamiento) y otra
    determinista (para el subconjunto de validación). Ambas apuntan a los
    mismos archivos; solo cambia la transformación que se aplica.
    """
    train_tf, eval_tf = get_transforms()

    train_full_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    train_full_eval = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

    targets = np.array(train_full_aug.targets)
    train_idx, val_idx = stratified_split(targets, val_size=val_size, seed=seed)

    # Verificación: entrenamiento y validación no comparten ninguna imagen.
    overlap = np.intersect1d(train_idx, val_idx)
    assert overlap.size == 0, f"Hay {overlap.size} índices repetidos entre train y val"

    train_set = Subset(train_full_aug, train_idx)
    val_set = Subset(train_full_eval, val_idx)

    # Generador con semilla 57 -> el orden del shuffle es reproducible.
    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(num_workers=num_workers, worker_init_fn=seed_worker,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=g, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)

    if verbose:
        print("=" * 60)
        print(f"PARTICIÓN ESTRATIFICADA (semilla = {seed}, validación = {val_size:.0%})")
        print("=" * 60)
        print(f"Entrenamiento oficial completo : {len(targets):>6} imágenes")
        print(f"  -> Subconjunto entrenamiento : {len(train_set):>6} imágenes")
        print(f"  -> Subconjunto validación    : {len(val_set):>6} imágenes")
        print(f"Prueba (independiente)         : {len(test_set):>6} imágenes")
        print(f"Índices solapados train/val    : {overlap.size}")
        print(f"Lotes por época -> train: {len(train_loader)}, "
              f"val: {len(val_loader)}, test: {len(test_loader)}")

    split_info = {"train_idx": train_idx, "val_idx": val_idx, "targets": targets}
    return train_loader, val_loader, test_loader, split_info


# ---------------------------------------------------------------------------
# 5. Visualización
# ---------------------------------------------------------------------------
def show_samples(dataset, n_images: int = 16, seed: int = SEED,
                 save_path: Path | None = None, show: bool = True):
    """
    Muestra una cuadrícula de n_images (entre 8 y 16) con su etiqueta real.
    Usa las imágenes originales (dataset.data, uint8 HWC) para verlas sin normalizar.
    """
    assert 8 <= n_images <= 16, "La guía pide entre 8 y 16 imágenes"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.data), size=n_images, replace=False)

    cols = 4
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(dataset.data[i])
        label = dataset.targets[i]
        ax.set_title(f"{CLASSES[label]} ({label})", fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[n_images:]:
        ax.axis("off")
    fig.suptitle(f"CIFAR-10 — {n_images} muestras con su etiqueta real (semilla {seed})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])   # deja el 5 % superior libre para el título
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def show_augmentations(dataset, index: int = 0, n_versions: int = 6,
                       save_path: Path | None = None, show: bool = True):
    """Aplica el pipeline de entrenamiento varias veces sobre la MISMA imagen."""
    train_tf, _ = get_transforms()
    img, label = dataset[index]           # imagen PIL original
    fig, axes = plt.subplots(1, n_versions + 1, figsize=(2.2 * (n_versions + 1), 2.6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for k in range(1, n_versions + 1):
        aug = denormalize(train_tf(img)).permute(1, 2, 0).numpy()  # CHW -> HWC
        axes[k].imshow(aug)
        axes[k].set_title(f"Aumentada {k}")
        axes[k].axis("off")
    fig.suptitle(f"Data Augmentation sobre una imagen de la clase '{CLASSES[label]}'",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])   # espacio para el título
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ejecución directa del script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Semilla individual aplicada: {SEED}")

    # Carga y exploración
    train_raw, test_raw = load_raw_datasets()
    print("\n" + "=" * 60)
    print("DIMENSIONES DEL DATASET ORIGINAL")
    print("=" * 60)
    print(f"Entrenamiento: {train_raw.data.shape}  (N, H, W, C), dtype={train_raw.data.dtype}")
    print(f"Prueba       : {test_raw.data.shape}")
    print(f"Clases ({len(train_raw.classes)}): {train_raw.classes}")

    print("\nDistribución de clases (entrenamiento oficial):")
    for name, count in class_distribution(train_raw.targets).items():
        print(f"  {name:<11} {count}")

    # Partición estratificada y DataLoaders
    train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=64)

    print("\nDistribución por clase después de la partición:")
    print(f"  {'clase':<11} {'train':>6} {'val':>6}")
    dist_train = class_distribution(info["targets"], info["train_idx"])
    dist_val = class_distribution(info["targets"], info["val_idx"])
    for name in CLASSES:
        print(f"  {name:<11} {dist_train[name]:>6} {dist_val[name]:>6}")

    # Comprobación del pipeline: forma del primer lote (tensor 4D)
    images, labels = next(iter(train_loader))
    print("\nPrimer lote del DataLoader de entrenamiento:")
    print(f"  images.shape = {tuple(images.shape)}  -> (Batch, Channels, Height, Width)")
    print(f"  labels.shape = {tuple(labels.shape)}")
    print(f"  rango de valores tras normalizar: [{images.min():.3f}, {images.max():.3f}]")

    # Visualización
    show_samples(train_raw, n_images=16, save_path=OUTPUT_DIR / "samples_grid.png")
    show_augmentations(train_raw, index=0, save_path=OUTPUT_DIR / "augmentation_examples.png")
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")