"""
model.py — Pregunta 3
Arquitectura CNN en PyTorch, conexión Flatten -> Linear con cálculo
dinámico de tamaño y validación del forward pass con tensores 4D.

Formato de tensores: NCHW = (Batch_Size, Channels, Height, Width)

Ejecución (desde la raíz del repositorio):
    python src/model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

SEED = 57
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)   # (C, H, W) de una imagen CIFAR-10


class CNN(nn.Module):
    """
    Bloque 1 : Conv2d(3->32, k=3, p=1)  -> ReLU -> MaxPool2d(2)   [B,3,32,32] -> [B,32,16,16]
    Bloque 2 : Conv2d(32->64, k=3, p=1) -> ReLU -> MaxPool2d(2)   [B,32,16,16] -> [B,64,8,8]
    Flatten  :                                                    [B,64,8,8]  -> [B,4096]
    Densa 1  : Linear(4096->128) -> ReLU -> Dropout(p)            [B,4096]    -> [B,128]
    Salida   : Linear(128->10)   (logits, SIN Softmax)            [B,128]     -> [B,10]
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()   # aplana desde la dimensión 1: [B,C,H,W] -> [B, C*H*W]

        # Cálculo DINÁMICO del tamaño de entrada a la primera capa Linear:
        # se pasa un tensor ficticio por los bloques convolucionales y se mide.
        self.flatten_size = self._get_flatten_size(INPUT_SHAPE)
        assert self.flatten_size == 64 * 8 * 8, "El tamaño aplanado debería ser 4096"

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),   # logits: CrossEntropyLoss aplica LogSoftmax internamente
        )

    @torch.no_grad()
    def _get_flatten_size(self, input_shape) -> int:
        dummy = torch.zeros(1, *input_shape)          # [1, 3, 32, 32]
        out = self.block2(self.block1(dummy))         # [1, 64, 8, 8]
        return int(out.numel())                       # 64 * 8 * 8 = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        x = self.block1(x)        # [B, 32, 16, 16]
        x = self.block2(x)        # [B, 64, 8, 8]
        x = self.flatten(x)       # [B, 4096]
        x = self.classifier(x)    # [B, 10]
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_forward_pass(model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """
    Pasa un lote SIMULADO por el modelo capa por capa, imprime los shapes
    intermedios y comprueba con aserciones que no exista Shape Mismatch.
    """
    model.eval()
    x = torch.randn(batch_size, *INPUT_SHAPE)   # lote simulado 4D
    assert x.dim() == 4, "La entrada debe ser un tensor 4D (B, C, H, W)"

    print("=" * 66)
    print("VALIDACIÓN DEL FORWARD PASS (lote simulado)")
    print("=" * 66)
    print(f"{'Etapa':<38}{'Shape de salida':<28}")
    print("-" * 66)
    print(f"{'Entrada':<38}{str(list(x.shape)):<28}")

    out = x
    with torch.no_grad():
        for block_name in ("block1", "block2"):
            for layer in getattr(model, block_name):
                out = layer(out)
                name = f"{block_name}.{layer.__class__.__name__}"
                print(f"{name:<38}{str(list(out.shape)):<28}")
        assert out.shape == (batch_size, 64, 8, 8), f"Shape inesperado antes de Flatten: {out.shape}"

        out = model.flatten(out)
        print(f"{'Flatten':<38}{str(list(out.shape)):<28}")
        assert out.shape == (batch_size, model.flatten_size)

        for layer in model.classifier:
            out = layer(out)
            print(f"{'classifier.' + layer.__class__.__name__:<38}{str(list(out.shape)):<28}")

        # El forward() completo debe dar exactamente lo mismo que el recorrido manual.
        logits = model(x)

    assert logits.shape == (batch_size, NUM_CLASSES), f"Salida incorrecta: {logits.shape}"
    assert torch.allclose(logits, out), "forward() no coincide con el recorrido capa por capa"
    print("-" * 66)
    print(f"Salida final (logits): {list(logits.shape)}  -> {NUM_CLASSES} clases")
    print("Aserciones superadas: el modelo procesa tensores 4D sin Shape Mismatch.")
    return logits


if __name__ == "__main__":
    torch.manual_seed(SEED)

    model = CNN()
    print(model)
    print(f"\nTamaño calculado dinámicamente para Flatten: {model.flatten_size}")
    print(f"Parámetros entrenables: {count_parameters(model):,}\n")

    validate_forward_pass(model, batch_size=8)

    # Comparación ilustrativa: primera capa de un MLP que recibe la imagen aplanada
    mlp_first_layer = nn.Linear(3 * 32 * 32, 1024)
    print(f"\nReferencia: una sola capa Linear(3072 -> 1024) de un MLP tiene "
          f"{count_parameters(mlp_first_layer):,} parámetros.")
