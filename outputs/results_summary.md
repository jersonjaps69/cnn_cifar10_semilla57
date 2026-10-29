# Resumen de resultados (semilla 57)

- Dispositivo: `cpu`
- Épocas: 20 | batch: 64 | lr: 0.001 | weight_decay: 0.0 | dropout: 0.5
- Parámetros entrenables: 545,098
- Tiempo total de entrenamiento: 7.9 min

## Métricas por época

| Época | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|---:|---:|---:|---:|---:|
| 1 | 1.7339 | 36.30 | 1.3930 | 49.50 |
| 2 | 1.4581 | 47.12 | 1.2365 | 55.31 |
| 3 | 1.3420 | 51.81 | 1.1184 | 60.35 |
| 4 | 1.2553 | 55.26 | 1.0220 | 63.28 |
| 5 | 1.2069 | 57.19 | 0.9824 | 65.15 |
| 6 | 1.1805 | 58.24 | 0.9480 | 66.53 |
| 7 | 1.1462 | 59.62 | 0.9412 | 66.79 |
| 8 | 1.1195 | 60.84 | 0.9242 | 67.66 |
| 9 | 1.0971 | 61.52 | 0.8995 | 68.57 |
| 10 | 1.0787 | 62.24 | 0.8767 | 69.10 |
| 11 | 1.0640 | 62.84 | 0.8518 | 69.80 |
| 12 | 1.0448 | 63.38 | 0.8812 | 68.80 |
| 13 | 1.0353 | 63.82 | 0.8220 | 71.22 |
| 14 | 1.0214 | 64.53 | 0.8248 | 70.77 |
| 15 | 1.0100 | 64.87 | 0.8163 | 71.57 |
| 16 | 0.9965 | 65.06 | 0.8230 | 71.40 |
| 17 | 0.9900 | 65.40 | 0.8005 | 71.98 |
| 18 | 0.9871 | 65.61 | 0.8119 | 71.58 |
| 19 **(mejor)** | 0.9642 | 66.60 | 0.7811 | 72.89 |
| 20 | 0.9573 | 66.52 | 0.7956 | 72.28 |

## Conjunto de prueba (10 000 imágenes, modelo de la mejor época)

- Test Loss: **0.7799**
- Test Accuracy: **73.09 %**

| Clase | Precision | Recall | F1-score |
|---|---:|---:|---:|
| airplane | 0.738 | 0.774 | 0.755 |
| automobile | 0.816 | 0.903 | 0.858 |
| bird | 0.675 | 0.548 | 0.605 |
| cat | 0.542 | 0.546 | 0.544 |
| deer | 0.690 | 0.671 | 0.680 |
| dog | 0.645 | 0.640 | 0.643 |
| frog | 0.751 | 0.813 | 0.781 |
| horse | 0.800 | 0.770 | 0.785 |
| ship | 0.823 | 0.838 | 0.831 |
| truck | 0.808 | 0.806 | 0.807 |

## Diagnóstico automático (orientativo)

- Brecha de accuracy train-val (última época): -5.76 puntos
- Brecha de accuracy train-val (mejor época): -6.29 puntos
- Val loss mínima: 0.7811 (época 19)
- Aumento de val loss desde su mínimo: 0.0145
- Síntomas de UNDERFITTING: el accuracy de entrenamiento sigue por debajo de 70 %; el modelo aún no se ajusta bien ni a los datos que ve.
- Validación > entrenamiento: es esperable con Data Augmentation y Dropout, porque el accuracy de entrenamiento se mide sobre imágenes alteradas y con neuronas apagadas, mientras la validación usa imágenes limpias y modo eval.
