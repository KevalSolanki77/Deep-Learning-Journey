# Day 19 - Basic CNN Architecture & LeNet-5

## The Shared Idea

A classical Convolutional Neural Network (CNN) is structured into two functional halves: a hierarchical **feature extractor** followed by a **classifier**.

```text
       Input Image
            │
            ▼
┌────────────────────────┐
│      Convolution       │ ◄── Learns local patterns (shared spatial weights)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Activation Function   │ ◄── Introduces non-linearity (Tanh, ReLU)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│     Pooling Layer      │ ◄── Downsamples spatial dimensions (Max / Avg)
└───────────┬────────────┘
            │
            ├─► [Repeated × N blocks: Feature Extraction Hierarchy]
            │
            ▼
┌────────────────────────┐
│ Flatten / Global Pool  │ ◄── Collapses 2D/3D maps to 1D representation
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│      Dense Layer       │ ◄── Fully connected mapping of high-level features
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│     Output Layer       │ ◄── Class probability distribution (Softmax)
└────────────────────────┘
```

* **Feature Extractor:** Repeated blocks of Convolution $\rightarrow$ Activation $\rightarrow$ Pooling.
  * Early layers capture primitive low-level signals (edges, corners, gradients, strokes).
  * Intermediate layers compose edges into motifs, textures, and shape fragments.
  * Deep layers synthesize motifs into high-level semantic object components.
* **Classifier:** Fully connected dense layers (or a Global Average Pool followed by a single linear layer) map the aggregated spatial descriptors to categorical logits.

### The Inverted Spatial-Channel Law

| Dimension | Direction | Structural Reason |
|---|---|---|
| **Height & Width ($H \times W$)** | Shrink ($28 \rightarrow 14 \rightarrow 5$) | Pooling and `valid` convolutions discard precise pixel coordinates in exchange for translational tolerance and computational efficiency. |
| **Channels / Filters ($C$)** | Grow ($6 \rightarrow 16 \rightarrow 120$) | Spatial downsampling frees up computational bandwidth, allowing the model to detect a broader dictionary of complex, abstract patterns. |

**Weight Sharing vs. Full Connectivity:** Convolutions slide identical kernels across the input, meaning their parameter counts depend strictly on kernel dimensions and channel depth:

$$\text{Params} = (k_h \cdot k_w \cdot C_{\text{in}} + 1) \cdot C_{\text{out}}$$

Dense layers offer zero spatial sharing; their weight count scales directly with the incoming flattened vector size:

$$\text{Params} = (N_{\text{in}} + 1) \cdot N_{\text{out}}$$

In early CNNs, this structural difference causes the dense classification head to swallow almost the entire parameter budget.

---

## LeNet-5 (LeCun et al., 1998)

Introduced by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner for handwritten check recognition at commercial banking scale. It established the modern blueprint for gradient-based end-to-end visual recognition.

### Classical Specification ($32 \times 32 \times 1$ Input)

Spatial downsampling follows the standard valid convolution formula:

$$\text{output} = \left\lfloor \frac{n - k}{s} \right\rfloor + 1$$

| Layer | Operation | Output Shape | Parameter Calculation | Total Params |
|---|---|---|---|---|
| **Input** | Grayscale Canvas | $(32, 32, 1)$ | — | 0 |
| **C1** | Conv2D, $6 \times (5 \times 5)$, valid, $s=1$ | $(28, 28, 6)$ | $(5 \cdot 5 \cdot 1 + 1) \cdot 6$ | 156 |
| **S2** | Average Pool $2 \times 2$, $s=2$ | $(14, 14, 6)$ | Non-parametric (fixed mean) | 0 |
| **C3** | Conv2D, $16 \times (5 \times 5)$, valid, $s=1$ | $(10, 10, 16)$ | $(5 \cdot 5 \cdot 6 + 1) \cdot 16$ | 2,416 |
| **S4** | Average Pool $2 \times 2$, $s=2$ | $(5, 5, 16)$ | Non-parametric (fixed mean) | 0 |
| **C5** | Conv2D (or Dense), $120 \times (5 \times 5)$ | $(1, 1, 120)$ | $(5 \cdot 5 \cdot 16 + 1) \cdot 120$ | 48,120 |
| **F6** | Dense | $(84,)$ | $(120 + 1) \cdot 84$ | 10,164 |
| **Output** | Dense (Softmax / Euclidean RBF) | $(10,)$ | $(84 + 1) \cdot 10$ | 850 |

**Total Parameters:** **61,706**

---

## Key Theoretical Insights

### 1. C5: When a Convolution Equals a Dense Layer
Layer C5 applies a $5 \times 5$ kernel to a feature map that has shrunk to exactly $5 \times 5$. Because the kernel matches the spatial input dimensions, the sliding window can only sit in one position:

$$(5 - 5) + 1 = 1 \times 1$$

Its parameter count is identical to a dense transition:

$$(5 \cdot 5 \cdot 16) \cdot 120 + 120 = 400 \cdot 120 + 120 = 48,120$$

Treating C5 as a convolution allows the network to process larger inputs (e.g., $64 \times 64$) without code changes, outputting a 2D grid of classification predictions instead of crashing. This equivalence directly inspired Fully Convolutional Networks (FCNs) for dense semantic segmentation.

### 2. Parameter Distribution
In LeNet-5, layer **C5 accounts for ~78.0%** of the model's total weight budget ($48,120 / 61,706$). In contrast, layer **C1 represents just ~0.25%** ($156 / 61,706$). Weight sharing keeps feature extraction lightweight, while dense projections dominate the memory footprint.

### 3. Receptive Field Unrolling
While C1 units observe only a local $5 \times 5$ window, hierarchical pooling and subsequent convolutions compound the receptive field. By layer C5, every activation scalar reflects contextual information integrated across the entire original $32 \times 32$ image.

### 4. The $28 \times 28$ Canvas Nuance (MNIST / Fashion-MNIST)
When feeding unpadded $28 \times 28$ inputs directly into LeNet's valid convolutions:

$$28 \xrightarrow{\text{C1 (valid)}} 24 \xrightarrow{\text{S2}} 12 \xrightarrow{\text{C3 (valid)}} 8 \xrightarrow{\text{S4}} 4$$

The feature map arriving at C5 is only $4 \times 4$. A true $5 \times 5$ convolution fails here with a negative dimension error ($k > n$). Implementing C5 as `Flatten()` $\rightarrow$ `Dense(120)` avoids this error by accepting the $4 \times 4 \times 16 = 256$ vector silently, resulting in:

$$(256 + 1) \cdot 120 = 30,840 \text{ parameters}$$

This reduces the total parameter count from **61,706** to **44,426**. To preserve the original 61,706-parameter layout on $28 \times 28$ data, C1 must use `padding="same"`.

---

## Code Implementations (Keras 3 / TensorFlow)

### Classical LeNet-5 Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet5_padded(input_shape=(28, 28, 1), num_classes=10):
    """
    Classical LeNet-5 using padding='same' on C1 to preserve
    the original 61,706-parameter count on 28x28 inputs.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # C1: Same padding preserves dimension to 28x28x6
        layers.Conv2D(6, kernel_size=5, strides=1, padding="same", activation="tanh", name="C1"),
        # S2: Average pooling downsamples to 14x14x6
        layers.AveragePooling2D(pool_size=2, strides=2, padding="valid", name="S2"),
        # C3: Valid convolution shrinks to 10x10x16
        layers.Conv2D(16, kernel_size=5, strides=1, padding="valid", activation="tanh", name="C3"),
        # S4: Average pooling downsamples to 5x5x16
        layers.AveragePooling2D(pool_size=2, strides=2, padding="valid", name="S4"),
        # C5: Fully connected dense projection
        layers.Flatten(name="Flatten"),
        layers.Dense(120, activation="tanh", name="C5_Dense"),
        # F6: Fully connected hidden layer
        layers.Dense(84, activation="tanh", name="F6"),
        # Output classification head
        layers.Dense(num_classes, activation="softmax", name="Output")
    ])
    return model

model = build_lenet5_padded()
model.summary()  # Exactly 61,706 parameters
```

---

## LeNet-5 vs. Modern CNN Practice

| Architectural Dimension | Classical LeNet-5 (1998) | Modern Standard (ResNet, ConvNeXt) | Design Rationale |
|---|---|---|---|
| **Activation Function** | Tanh / Sigmoid | ReLU, Leaky ReLU, GELU | Prevents gradient saturation across deep backpropagation chains. |
| **Spatial Pooling** | Average Pooling | Max Pooling (internal), GAP (terminal) | Max pooling retains peak activations; GAP replaces dense parameters. |
| **Internal Normalization** | None | Batch Normalization, Layer Normalization | Smooths optimization landscapes; stabilizes internal covariate shifts. |
| **Regularization** | Weight Decay ($L_2$) | Dropout, DropPath, Data Augmentation | Breaks co-adaptation of features; prevents memorization. |
| **Classifier Head** | Dense stack (`Flatten` $\rightarrow$ FC $\rightarrow$ FC) | Global Average Pooling (`GAP`) $\rightarrow$ Dense | Replaces millions of dense weights with a single spatial average per channel. |
| **Network Depth** | 5 learned weight layers | 50 to 1000+ layers | Enabled by residual skip connections ($x + \mathcal{F}(x)$) and normalization. |
| **Model Capacity** | $\approx 6.1 \times 10^4$ parameters | $10^7$ to $10^9+$ parameters | Scaled up for complex, unconstrained RGB photography. |

---

## Empirical Ablation Study (Fashion-MNIST)

All variants were trained on Fashion-MNIST ($28 \times 28 \times 1$) using identical hyperparameters: Adam optimizer, learning rate default, batch size 128, 10 epochs, 10% validation split.

| Variant | Modifications | Test Accuracy | Total Parameters | Primary Takeaway |
|---|---|---|---|---|
| **LeNet-5 (Baseline)** | Valid convs, Tanh, Average Pooling | 87.43% | 44,426 | Baseline performance; slower convergence due to saturating Tanh. |
| **+ ReLU Activation** | Tanh $\rightarrow$ ReLU throughout | 86.86% | 44,426 | Slightly lower on Fashion-MNIST without tuning, but converges faster without vanishing gradients. |
| **+ Max Pooling** | AvgPool $\rightarrow$ MaxPool ($2 \times 2, s=2$) | **88.90%** | 44,426 | **+2.04% jump.** Routing gradients strictly to peak activations sharpens silhouette edge extraction. |
| **+ Batch Normalization** | Added BN after C1 and C3 | **89.03%** | 44,514 | Stabilizes layer inputs; training accuracy reached >93% by epoch 8. |
| **+ Dropout (0.25)** | 25% dropout prior to final Dense | **89.53%** | 44,514 | **Best performance.** Lowers test loss to 0.3039 by preventing co-adaptation in the classifier head. |
| **+ GAP Head** | Replaced `Flatten` + Dense with GAP | 81.25% | **4,894** | **89% parameter drop.** Performance drops here because compressing $4 \times 4 \times 16$ into 16 channels creates an informational bottleneck. |

---

## Failure Modes & Limitations

* **Small, Clean Image Constraint:** Designed for centered, normalized grayscale digits. Fails on cluttered scenes, complex backgrounds, and high-resolution color inputs.
* **Saturating Tanh Gradients:** Gradients scale as $(1 - \tanh^2(x))$; stacking more than 3-4 Tanh layers drives gradients to near zero, stalling early feature updates.
* **Lack of Regularization:** Prone to overfitting without dropout, data augmentation, or normalization.
* **Sensitivity to Translations & Scale:** Lacks multi-scale feature pyramids or modern data augmentation, leaving it brittle to off-center shifts and scale variations.

---

## Rules of Thumb

1. **Use ReLU as Default:** Replace Tanh with ReLU or its variants in every layer except recurrent gates or bounded output heads.
2. **Max Pool Inside, GAP at the Exit:** Use max pooling inside feature extractors to capture dominant boundaries, and Global Average Pooling before the final dense layer to control parameter growth.
3. **Normalize Early and Often:** Insert Batch Normalization (or Layer Normalization) after convolutions to accelerate convergence and allow higher learning rates.
4. **Beware GAP on Narrow Channels:** Global Average Pooling works best when feature extractors output wide channel depths (e.g., 256, 512). Collapsing shallow channel maps (e.g., 16 channels) discards too much information.