# Day 17 - Convolutional Padding and Strides

## The Shared Idea

Padding and strides govern the spatial dimension transformations of feature maps in convolutional neural networks. 

- **Padding** controls border accessibility, determining whether edge pixels receive equal operational focus and whether spatial resolution is preserved or degraded.
- **Stride** defines the step size across spatial dimensions, dictating the downsampling rate and the rate of receptive field expansion.

Crucially, **neither parameter changes the number of trainable weights**. The parameter count of a standard 2D convolution depends strictly on kernel geometry and channel depth:

$$\text{Params} = \left( k_h \times k_w \times C_{\text{in}} + 1 \right) \times C_{\text{out}}$$

However, both alter the arithmetic computational footprint (FLOPs) and subsequent layer memory, because runtime operations scale directly with spatial output resolution:

$$\mathcal{O}\left(H_{\text{out}} \times W_{\text{out}} \times k_h \times k_w \times C_{\text{in}} \times C_{\text{out}}\right)$$

| Setting | Mechanism | Primary Utility | Parameter Impact | FLOPs Impact |
|---|---|---|---|---|
| **Padding (`valid`)** | No artificial boundary ($p=0$) | Natural signal processing; eliminates zero-bias | None | Base cost |
| **Padding (`same`)** | Symmetric/asymmetric boundary zero-fill | Preserves spatial resolution; deep feature preservation | None | Higher (larger output map) |
| **Stride ($s = 1$)** | Overlapping, single-pixel shift | High-resolution spatial extraction | None | Baseline dense sweep |
| **Stride ($s > 1$)** | Skips $s-1$ pixels per step | Strided downsampling; receptive field expansion | None | Drastically lower ($\approx 1/s^2$) |

---

## Output Dimension Formulas

Let $n$ denote input dimension (height or width), $k$ kernel dimension, $p$ total padding on both sides ($p = p_{\text{start}} + p_{\text{end}}$), and $s$ stride.

### General Form (Explicit Padding)

$$\text{output} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

### Framework Shortcuts (Keras / TensorFlow)

* **Valid (`padding="valid"`, $p = 0$):**
  $$\text{output}_{\text{valid}} = \left\lfloor \frac{n - k}{s} \right\rfloor + 1$$

* **Same (`padding="same"`):**
  $$\text{output}_{\text{same}} = \left\lceil \frac{n}{s} \right\rceil$$

For `padding="same"` with $s=1$, the required total padding across both sides satisfies:

$$p_{\text{total}} = k - 1 \implies p_{\text{side}} = \frac{k - 1}{2}$$

This demonstrates why odd kernel dimensions ($3\times3$, $5\times5$, $7\times7$) are ubiquitous: an odd $k$ yields an integer padding per side ($p=1$ for $3\times3$, $p=2$ for $5\times5$), preventing asymmetric boundary artifacts.

### Worked Dimensional Checks

* $n=5$, $k=3$, $p=0$, $s=1$:
  $$\left\lfloor \frac{5 + 0 - 3}{1} \right\rfloor + 1 = 2 + 1 = 3$$
* $n=224$, $k=3$, `same`, $s=2$:
  $$\left\lceil \frac{224}{2} \right\rceil = 112$$
* $n=224$, $k=7$, $p=3$ per side ($2p = 6$), $s=2$:
  $$\left\lfloor \frac{224 + 6 - 7}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = 112$$

---

## Code Implementations & Structural Comparison

Comparison of three identical 3-layer feedforward convolutional stacks operating on $(28, 28, 1)$ inputs under varying padding and stride configurations:

```python
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten

# Configuration A: Valid Padding (Shrinks each layer: 28 -> 26 -> 24 -> 22)
model_valid = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Configuration B: Same Padding with s=1 (Preserves spatial resolution: 28 -> 28 -> 28 -> 28)
model_same = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Configuration C: Same Padding with s=2 (Downsamples aggressively: 28 -> 14 -> 7 -> 4)
model_strided = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Architectural Impact Summary

The choice of padding and strides directly dictates downstream parameter distribution:

| Metric | Config A (`valid`, $s=1$) | Config B (`same`, $s=1$) | Config C (`same`, $s=2$) |
|---|---|---|---|
| **Conv Layer 1 Shape** | $(26, 26, 32)$ | $(28, 28, 32)$ | $(14, 14, 32)$ |
| **Conv Layer 2 Shape** | $(24, 24, 32)$ | $(28, 28, 32)$ | $(7, 7, 32)$ |
| **Conv Layer 3 Shape** | $(22, 22, 32)$ | $(28, 28, 32)$ | $(4, 4, 32)$ |
| **Flatten Vector Dim** | $15,488$ | $25,088$ | **$512$** |
| **First Dense Weights** | $1,982,592$ | $3,211,392$ | **$65,664$** |
| **Total Model Params** | $2,002,698$ | $3,231,498$ | **$85,770$** |

> **Takeaway:** Using strided convolutions reduces the dense transition vector from $25,088$ elements down to $512$, slashing model parameters from **3.2M to 85k** without changing convolutional filter parameters.

---

## Edge Cases, Framework Differences, and Failure Modes

### 1. Asymmetric Padding in TensorFlow vs. PyTorch
When total padding $p_{\text{total}} = n_{\text{out}} \times s - n + k - 1$ is an odd integer, symmetric division is impossible:
* **TensorFlow / Keras:** Allocates the extra zero to the **bottom and right** borders:
  $$p_{\text{top}} = \lfloor p / 2 \rfloor, \quad p_{\text{bottom}} = p - p_{\text{top}}$$
* **PyTorch:** Requires explicit symmetric margins `padding=(p, p)`. If given an odd integer manual specification, it pads symmetrically unless wrapped with `nn.ZeroPad2d((left, right, top, bottom))`. Porting weights between frameworks without accounting for this produces subtle boundary distribution divergence.

### 2. Silent Truncation via `floor()`
If $(n + 2p - k)$ is not cleanly divisible by $s$, the trailing pixels fail to form a full receptive window and are dropped silently without raising exceptions:

$$\text{Dropped Pixels} = (n + 2p - k) \pmod s$$

In residual architectures (e.g., ResNet skip connections), an untracked floor operation will result in an element-wise addition mismatch between identity branch $x$ and residual branch $\mathcal{F}(x)$.

### 3. Artificial Boundary Bias
Standard zero padding introduces artificial low-magnitude borders into feature maps. In high-frequency tasks (edge detection, image restoration, or audio spectrogram processing), zero borders introduce artificial step-discontinuities. Where boundaries carry critical phase or continuity data, reach for alternative padding strategies:
* **Reflect Padding:** Mirrors interior pixels across boundaries (`tf.pad(..., mode='REFLECT')`).
* **Replication Padding:** Clamps and extends border pixels outward.

---

## Rules of Thumb

- **Resolution Maintenance:** Standardize on `padding="same"` with stride $s=1$ for feature extractors where spatial degradation needs to be isolated from depth (e.g., U-Net encoder-decoder stages, dense object detection heads).
- **Compute Reduction:** Prefer strided convolution ($s=2$) over standard pooling (MaxPool/AvgPool) when you want the network to learn its own downsampling kernel rather than relying on a fixed heuristic.
- **Kernel Symmetry:** Stick strictly to odd kernel sizes ($3\times3$, $5\times5$) to ensure identical integer border padding on all sides.
- **Memory Optimization:** In classification networks, apply aggressive striding ($s=2$) in the earliest layers (as in ResNet-50's initial $7\times7, s=2$ stage) to cut memory consumption before expanding channel width.