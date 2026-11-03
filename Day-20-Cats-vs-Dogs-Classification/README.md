# Day 20 - Milestone Project: Dogs vs. Cats Classification

## Project Overview

This milestone project implements an end-to-end binary image classifier using a custom Convolutional Neural Network (CNN) trained on the Microsoft Cats vs. Dogs dataset (25,000 real-world color photographs). 

The goal of this milestone is to construct a production-style input pipeline capable of parsing diverse, noisy image data, handling corrupted JPEG headers in flight, stabilizing gradient updates across deep convolutional blocks via Batch Normalization, and mitigating overfitting using systematic Dropout regularization.

---

## Dataset Ingestion & Preprocessing Pipeline

The project uses the `PetImages` dataset structured into two class subdirectories: `Cat/` and `Dog/`.

```text
PetImages/
├── Cat/  (12,500 images: 0.jpg, 1.jpg, ...)
└── Dog/  (12,500 images: 0.jpg, 1.jpg, ...)
```

### 1. Ingestion via `image_dataset_from_directory`
Instead of holding 25,000 raw color images in memory, data is streamed dynamically via generator batches using an 80/20 train-to-validation split:

* **Training Set:** 20,000 images (`subset="training"`)
* **Validation Set:** 5,000 images (`subset="validation"`)
* **Resolution:** $256 \times 256 \times 3$ RGB
* **Batch Size:** 100
* **Label Representation:** Binary scalar integers ($0 = \text{Cat}$, $1 = \text{Dog}$)

### 2. Normalization & Fault Tolerance
* **Channel Rescaling:** Image pixel intensities are mapped from integer range $[0, 255]$ to floating-point representation $[0.0, 1.0]$ using vector mapping:
  $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{255.0}$$
* **Corrupt JPEG Fault Tolerance (`ignore_errors()`):** Real-world collections frequently contain damaged headers, invalid JFIF tags, and truncated byte markers (e.g., `extraneous bytes before marker 0xd9`). Chaining `dataset.ignore_errors()` prevents worker crash-outs by dropping invalid files during stream decoding.

---

## Model Architecture

The model combines a 3-stage convolutional feature extractor with a deeply regularized multi-layer perceptron classification head.

```text
                  Input: (256, 256, 3)
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: Conv2D(32, 3x3) ──> BN ──> MaxPool2D(2x2)  │ ──> Dropout(0.3)
└──────────────────────────┬──────────────────────────┘
                           │ Output: (127, 127, 32)
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Conv2D(64, 3x3) ──> BN ──> MaxPool2D(2x2)  │ ──> Dropout(0.3)
└──────────────────────────┬──────────────────────────┘
                           │ Output: (62, 62, 64)
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: Conv2D(128, 3x3) ──> BN ──> MaxPool2D(2x2) │ ──> Dropout(0.3)
└──────────────────────────┬──────────────────────────┘
                           │ Output: (30, 30, 128)
                           ▼
┌─────────────────────────────────────────────────────┐
│        Flatten (115,200)                            │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Dense(128) ──> BN ──> Dropout(0.3)                  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Dense(64)  ──> BN ──> Dropout(0.3)                  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Dense(1, Sigmoid) ──> Binary Output [0, 1]          │
└─────────────────────────────────────────────────────┘
```

### Layer-by-Layer Parameter Profile

| Layer (type) | Output Shape | Kernel / Stride | Param # | Role in Graph |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `(None, 256, 256, 3)` | — | $0$ | Input canvas |
| **Conv2D** | `(None, 254, 254, 32)` | $3 \times 3$, valid | $896$ | Early edge & stroke detection |
| **BatchNormalization** | `(None, 254, 254, 32)` | — | $128$ | Feature stabilization |
| **MaxPooling2D** | `(None, 127, 127, 32)` | $2 \times 2$, stride 2 | $0$ | Spatial downsampling |
| **Dropout** | `(None, 127, 127, 32)` | rate $= 0.3$ | $0$ | Spatial feature regularization |
| **Conv2D** | `(None, 125, 125, 64)` | $3 \times 3$, valid | $18,496$ | Motif & texture detection |
| **BatchNormalization** | `(None, 125, 125, 64)` | — | $256$ | Feature stabilization |
| **MaxPooling2D** | `(None, 62, 62, 64)` | $2 \times 2$, stride 2 | $0$ | Spatial downsampling |
| **Dropout** | `(None, 62, 62, 64)` | rate $= 0.3$ | $0$ | Feature regularization |
| **Conv2D** | `(None, 60, 60, 128)` | $3 \times 3$, valid | $73,856$ | High-level component extraction |
| **BatchNormalization** | `(None, 60, 60, 128)` | — | $512$ | Feature stabilization |
| **MaxPooling2D** | `(None, 30, 30, 128)` | $2 \times 2$, stride 2 | $0$ | Spatial downsampling |
| **Dropout** | `(None, 30, 30, 128)` | rate $= 0.3$ | $0$ | Feature regularization |
| **Flatten** | `(None, 115200)` | — | $0$ | Transition unroll ($30 \times 30 \times 128$) |
| **Dense** | `(None, 128)` | ReLU | $14,745,728$ | Primary feature projection |
| **BatchNormalization** | `(None, 128)` | — | $512$ | Dense layer normalization |
| **Dropout** | `(None, 128)` | rate $= 0.3$ | $0$ | Co-adaptation prevention |
| **Dense** | `(None, 64)` | ReLU | $8,256$ | Hidden representation |
| **BatchNormalization** | `(None, 64)` | — | $256$ | Dense layer normalization |
| **Dropout** | `(None, 64)` | rate $= 0.3$ | $0$ | Co-adaptation prevention |
| **Dense (Output)** | `(None, 1)` | Sigmoid | $65$ | Probability score $\hat{y} \in [0, 1]$ |

* **Total Parameters:** $14,848,961$ ($56.64\text{ MB}$)
* **Trainable Parameters:** $14,848,129$ ($56.64\text{ MB}$)
* **Non-trainable Parameters:** $832$ ($3.25\text{ KB}$ from Batch Normalization moving stats)

> **Key Architectural Takeaway:** Notice that the transition dense layer (`Flatten` $\to$ `Dense(128)`) accounts for **$14,745,728$ parameters**, representing **99.3% of the total model budget**. In Day 18 and 19, we observed that replacing `Flatten()` with a `GlobalAveragePooling2D()` layer collapses $(30, 30, 128) \to (128,)$, reducing dense transition parameters from 14.7 million down to approximately 16,500.

---

## Experimental Code (TensorFlow / Keras)

```python
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Dataset loading with train/validation split
data_path = "/kaggle/input/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/PetImages"

train_ds = keras.utils.image_dataset_from_directory(
    data_path,
    labels="inferred",
    label_mode="int",
    image_size=(256, 256),
    batch_size=100,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    data_path,
    labels="inferred",
    label_mode="int",
    image_size=(256, 256),
    batch_size=100,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# 2. Rescaling and stream error suppression
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process).ignore_errors()
val_ds = val_ds.map(process).ignore_errors()

# 3. Model construction
model = keras.Sequential([
    layers.Input(shape=(256, 256, 3)),
    
    # Conv Block 1
    layers.Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.3),

    # Conv Block 2
    layers.Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.3),

    # Conv Block 3
    layers.Conv2D(128, kernel_size=(3, 3), padding="valid", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.3),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# 4. Compilation & Callbacks
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=2,
    factor=0.5,
    min_lr=0.00001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction]
)
```

---

## Training Diagnostics & Run Analysis

The model was trained for 10 epochs on an NVIDIA Tesla T4 GPU on Kaggle.

### Epoch Performance Summary

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 63.04% | 0.6867 | 53.62% | 0.6928 | $1 \times 10^{-3}$ |
| **2** | 71.75% | 0.5509 | 51.15% | 0.8665 | $1 \times 10^{-3}$ |
| **3** | 76.74% | 0.4832 | 59.60% | 0.7585 | $1 \times 10^{-3}$ |
| **4** | 80.29% | 0.4227 | 71.56% | 0.5555 | $1 \times 10^{-3}$ |
| **5** | 84.25% | 0.3578 | 78.62% | 0.4435 | $1 \times 10^{-3}$ |
| **6** | 87.76% | 0.2842 | 80.92% | 0.4328 | $1 \times 10^{-3}$ |
| **7** | 90.72% | 0.2229 | 78.85% | 0.4748 | $1 \times 10^{-3}$ |
| **8** | 93.16% | 0.1698 | 80.62% | 0.5041 | $1 \times 10^{-3}$ |
| **9** | **95.04%** | **0.1267** | **82.38%** | 0.5127 | $1 \times 10^{-3}$ |
| **10** | *Converging* | *Diverging* | — | — | *Plateau / Stop* |

### Overfitting & Generalization Diagnosis

```text
Training Curves:
Train Accuracy: 63.0%  ────────>  95.0% (Strong upward trajectory)
Val Accuracy:   53.6%  ────────>  82.4% (Plateau around Epoch 6-9)

Train Loss:     0.686  ────────>  0.126 (Monotonic decline)
Val Loss:       0.692  ──>  0.432 (Epoch 6 minimum) ──>  0.512 (Rising after Epoch 6)
```

1. **The Overfitting Gap:** By Epoch 9, training accuracy climbs to **95.04%** with a loss of **0.1267**, whereas validation accuracy plateaus near **82.38%** and validation loss rebounds from **0.4328 to 0.5127**.
2. **Root Cause:** Despite using a 30% Dropout rate throughout the network, the unpooled $115,200 \times 128$ dense projection layer contains over 14.7 million parameters. The dense head memorizes fine pixel configurations in the training partition.
3. **Best Checkpoint:** `EarlyStopping(restore_best_weights=True)` preserves the weights from **Epoch 6**, where validation loss reached its global minimum of **0.4328** with **80.92%** validation accuracy.

---

## Production Enhancements for Day 21+

1. **Replace `Flatten()` with `GlobalAveragePooling2D()`:** 
   Eliminates the 14.7M parameter dense transition, lowering parameter count to under 150k and directly curbing classifier memorization.
2. **Real-time Data Augmentation:**
   Introduce random horizontal flips, rotations, and zooms before the first convolution:
   ```python
   layers.RandomFlip("horizontal"),
   layers.RandomRotation(0.1),
   layers.RandomZoom(0.1)
   ```
3. **Transfer Learning Transition:**
   For high-resolution natural images, replacing custom 3-block CNNs with pre-trained feature extractors (e.g., MobileNetV2, ResNet50V2 initialized on ImageNet) boosts accuracy from **82% to 96%+** within 3–5 fine-tuning epochs.