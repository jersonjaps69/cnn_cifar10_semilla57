"""
predict.py — Pruebas visuales del modelo entrenado (complemento de la Pregunta 4)

Carga los mejores pesos (outputs/best_model.pth, generado por train.py),
clasifica imágenes del conjunto de PRUEBA y guarda:
  - outputs/test_predictions.png : 16 imágenes de prueba con clase real vs. predicha
  - outputs/test_errors.png      : 16 imágenes mal clasificadas (confusiones del modelo)

Ejecución (desde la raíz del repositorio, después de train.py):
    python src/predict.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # solo guarda las figuras, no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from dataset import CLASSES, DATA_DIR, OUTPUT_DIR, SEED, denormalize, get_transforms, set_seed
from model import CNN


def plot_grid(images, labels, preds, probs, title, path):
    """Cuadrícula 4x4: título verde si acierta, rojo si se equivoca."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 11))
    for k, ax in enumerate(axes.flat):
        if k >= len(images):
            ax.axis("off")
            continue
        ax.imshow(denormalize(images[k]).permute(1, 2, 0).numpy())   # CHW -> HWC
        ok = labels[k] == preds[k]
        ax.set_title(f"Real: {CLASSES[labels[k]]}\nPred: {CLASSES[preds[k]]} ({probs[k]:.0%})",
                     fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    set_seed(SEED)
    weights = OUTPUT_DIR / "best_model.pth"
    if not weights.exists():
        raise FileNotFoundError("No existe outputs/best_model.pth. Ejecute primero: python src/train.py")

    model = CNN()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    _, eval_tf = get_transforms()
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

    # Predicción sobre las 10 000 imágenes de prueba (en lotes)
    all_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    all_y = np.array(test_set.targets)
    probs_all = torch.cat([torch.softmax(model(all_x[i:i + 500]), dim=1)
                           for i in range(0, len(all_x), 500)])
    conf, preds = probs_all.max(dim=1)
    preds, conf = preds.numpy(), conf.numpy()
    print(f"Accuracy en prueba (verificación): {100 * (preds == all_y).mean():.2f}%")

    rng = np.random.default_rng(SEED)

    # 1) Muestra aleatoria de 16 imágenes de prueba
    idx = rng.choice(len(test_set), size=16, replace=False)
    hits = int((preds[idx] == all_y[idx]).sum())
    plot_grid(all_x[idx], all_y[idx], preds[idx], conf[idx],
              f"Predicciones en el conjunto de prueba — {hits}/16 aciertos (semilla {SEED})",
              OUTPUT_DIR / "test_predictions.png")

    # 2) 16 errores del modelo (confusiones)
    err_idx = np.where(preds != all_y)[0]
    sel = rng.choice(err_idx, size=min(16, len(err_idx)), replace=False)
    plot_grid(all_x[sel], all_y[sel], preds[sel], conf[sel],
              f"Ejemplos de imágenes mal clasificadas — {len(err_idx)} errores en 10 000",
              OUTPUT_DIR / "test_errors.png")

    # Confusiones más frecuentes (pares real -> predicha)
    pairs = {}
    for r, p in zip(all_y[err_idx], preds[err_idx]):
        pairs[(r, p)] = pairs.get((r, p), 0) + 1
    print("\nConfusiones más frecuentes (real -> predicha):")
    for (r, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {CLASSES[r]:<11} -> {CLASSES[p]:<11} {n} casos")

    print(f"\nFiguras guardadas en {OUTPUT_DIR}:")
    print("  - test_predictions.png\n  - test_errors.png")


if __name__ == "__main__":
    main()