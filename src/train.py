"""
train.py — Pregunta 4
Entrenamiento con CrossEntropyLoss + Adam, registro de Loss/Accuracy por
época (train y validación), evaluación en el conjunto de prueba
independiente y generación de gráficas en outputs/.

Ejecución (desde la raíz del repositorio):
    python src/train.py                 # valores por defecto
    python src/train.py --epochs 20     # cambiar hiperparámetros
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")   # guarda las figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASSES, OUTPUT_DIR, SEED, get_dataloaders, set_seed
from model import CNN, count_parameters


# ---------------------------------------------------------------------------
# Hiperparámetros configurables
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento CNN en CIFAR-10 (semilla 57)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje de Adam")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0, help="0 es lo más seguro en Windows")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Una época de entrenamiento
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                       # activa Dropout
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # 1. limpiar gradientes anteriores
        logits = model(images)          # 2. forward: [B,3,32,32] -> [B,10]
        loss = criterion(logits, labels)  # 3. pérdida (promedio del lote)
        loss.backward()                 # 4. backpropagation
        optimizer.step()                # 5. actualizar pesos (Adam)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Evaluación (validación o prueba)
# ---------------------------------------------------------------------------
@torch.no_grad()                        # no se calculan gradientes
def evaluate(model, loader, criterion, device):
    model.eval()                        # desactiva Dropout
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return (running_loss / total, 100.0 * correct / total,
            torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy())


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def plot_curves(history, best_epoch, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, history["train_loss"], "o-", label="Entrenamiento")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validación")
    ax1.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Mejor época ({best_epoch})")
    ax1.set_title("Curva de pérdida (CrossEntropyLoss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="Entrenamiento")
    ax2.plot(epochs, history["val_acc"], "s-", label="Validación")
    ax2.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Mejor época ({best_epoch})")
    ax2.set_title("Curva de exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f"Curvas de aprendizaje — CNN CIFAR-10 (semilla {SEED})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path, test_acc):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión — conjunto de prueba (accuracy {test_acc:.2f}%, semilla {SEED})")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


# ---------------------------------------------------------------------------
# Diagnóstico cuantitativo de overfitting / underfitting
# ---------------------------------------------------------------------------
def diagnose(history, best_epoch):
    """Criterios orientativos (no son reglas absolutas) para apoyar el análisis del README."""
    last = len(history["train_loss"]) - 1
    b = best_epoch - 1
    gap_acc_last = history["train_acc"][last] - history["val_acc"][last]
    gap_acc_best = history["train_acc"][b] - history["val_acc"][b]
    min_val_loss = min(history["val_loss"])
    min_val_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    val_loss_rise = history["val_loss"][last] - min_val_loss

    notes = []
    if gap_acc_last > 10 and val_loss_rise > 0.1:
        notes.append("Síntomas de OVERFITTING: la brecha train-val supera 10 puntos y la "
                     "pérdida de validación subió desde su mínimo.")
    elif val_loss_rise > 0.1:
        notes.append("Inicio de OVERFITTING: la pérdida de validación aumenta desde su mínimo "
                     "aunque la brecha de accuracy todavía es moderada.")
    if history["train_acc"][last] < 70:
        notes.append("Síntomas de UNDERFITTING: el accuracy de entrenamiento sigue por debajo "
                     "de 70 %; el modelo aún no se ajusta bien ni a los datos que ve.")
    if gap_acc_last < 0:
        notes.append("Validación > entrenamiento: es esperable con Data Augmentation y Dropout, "
                     "porque el accuracy de entrenamiento se mide sobre imágenes alteradas y con "
                     "neuronas apagadas, mientras la validación usa imágenes limpias y modo eval.")
    if not notes:
        notes.append("Ajuste razonable: brecha pequeña y la pérdida de validación no muestra "
                     "un aumento sostenido.")

    return {
        "gap_acc_last_epoch": round(gap_acc_last, 2),
        "gap_acc_best_epoch": round(gap_acc_best, 2),
        "min_val_loss": round(min_val_loss, 4),
        "min_val_loss_epoch": min_val_loss_epoch,
        "val_loss_rise_from_min": round(val_loss_rise, 4),
        "notes": notes,
    }


def write_results_summary(path, args, history, best_epoch, test_loss, test_acc,
                          report_dict, diag, n_params, device, elapsed):
    """Genera outputs/results_summary.md con las tablas listas para copiar al README."""
    lines = [
        f"# Resumen de resultados (semilla {SEED})", "",
        f"- Dispositivo: `{device}`",
        f"- Épocas: {args.epochs} | batch: {args.batch_size} | lr: {args.lr} | "
        f"weight_decay: {args.weight_decay} | dropout: {args.dropout}",
        f"- Parámetros entrenables: {n_params:,}",
        f"- Tiempo total de entrenamiento: {elapsed / 60:.1f} min", "",
        "## Métricas por época", "",
        "| Época | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for e in range(len(history["train_loss"])):
        mark = " **(mejor)**" if e + 1 == best_epoch else ""
        lines.append(f"| {e + 1}{mark} | {history['train_loss'][e]:.4f} | "
                     f"{history['train_acc'][e]:.2f} | {history['val_loss'][e]:.4f} | "
                     f"{history['val_acc'][e]:.2f} |")
    lines += ["", "## Conjunto de prueba (10 000 imágenes, modelo de la mejor época)", "",
              f"- Test Loss: **{test_loss:.4f}**",
              f"- Test Accuracy: **{test_acc:.2f} %**", "",
              "| Clase | Precision | Recall | F1-score |", "|---|---:|---:|---:|"]
    for c in CLASSES:
        r = report_dict[c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} |")
    lines += ["", "## Diagnóstico automático (orientativo)", "",
              f"- Brecha de accuracy train-val (última época): {diag['gap_acc_last_epoch']} puntos",
              f"- Brecha de accuracy train-val (mejor época): {diag['gap_acc_best_epoch']} puntos",
              f"- Val loss mínima: {diag['min_val_loss']} (época {diag['min_val_loss_epoch']})",
              f"- Aumento de val loss desde su mínimo: {diag['val_loss_rise_from_min']}"]
    lines += [f"- {n}" for n in diag["notes"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Semilla: {SEED}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size, val_size=args.val_size, num_workers=args.num_workers)

    model = CNN(dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"Parámetros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, 0, None

    EPOCHS = args.epochs
    print(f"\n{'Época':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Tiempo':>6}")
    print("-" * 64)
    start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        flag = ""
        if va_acc > best_val_acc:     # guardar el mejor modelo según validación
            best_val_acc, best_epoch = va_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            flag = "  *"
        print(f"{epoch + 1:>5} | {tr_loss:>10.4f} | {tr_acc:>8.2f}% | "
              f"{va_loss:>8.4f} | {va_acc:>6.2f}% | {time.time() - t0:>5.1f}s{flag}")
    elapsed = time.time() - start
    print(f"\nEntrenamiento terminado en {elapsed / 60:.1f} min. "
          f"Mejor época: {best_epoch} (val acc {best_val_acc:.2f}%)")

    # Evaluación final en el conjunto de prueba independiente con el mejor modelo
    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / "best_model.pth")
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\nCONJUNTO DE PRUEBA -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    report_txt = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES,
                                        digits=4, output_dict=True)
    print("\nReporte de clasificación (prueba):\n" + report_txt)

    # Artefactos gráficos obligatorios
    plot_curves(history, best_epoch, OUTPUT_DIR / "loss_accuracy.png")
    cm = plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR / "confusion_matrix.png", test_acc)

    # Diagnóstico y archivos de apoyo para el README
    diag = diagnose(history, best_epoch)
    print("Diagnóstico orientativo:")
    for n in diag["notes"]:
        print(f"  - {n}")

    with open(OUTPUT_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for e in range(EPOCHS):
            w.writerow([e + 1, history["train_loss"][e], history["train_acc"][e],
                        history["val_loss"][e], history["val_acc"][e]])

    (OUTPUT_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    metrics = {
        "seed": SEED, "device": str(device), "hyperparameters": vars(args),
        "trainable_parameters": n_params, "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 2), "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 2), "diagnosis": diag,
        "confusion_matrix": cm.tolist(), "training_minutes": round(elapsed / 60, 2),
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_results_summary(OUTPUT_DIR / "results_summary.md", args, history, best_epoch,
                          test_loss, test_acc, report_dict, diag, n_params, device, elapsed)

    print(f"\nArchivos generados en {OUTPUT_DIR}:")
    for name in ("loss_accuracy.png", "confusion_matrix.png", "results_summary.md",
                 "metrics.json", "history.csv", "classification_report.txt"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
