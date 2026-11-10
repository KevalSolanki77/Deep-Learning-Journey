# Day 28: Deep RNNs & Bidirectional RNNs

A comprehensive theoretical and practical guide covering hierarchical sequential representations through **Deep (Stacked) RNNs** and full temporal context modeling through **Bidirectional RNNs (Bi-RNNs)**, featuring comparative benchmark architectures implemented in TensorFlow and Keras.

---

## Table of Contents
1. [Deep Recurrent Neural Networks (Deep RNNs)](#1-deep-recurrent-neural-networks-deep-rnns)
   * [Definition](#definition)
   * [Why We Need Vertical Depth](#why-we-need-vertical-depth)
   * [The 2D Computational Grid](#the-2d-computational-grid)
   * [Strengths & Limitations](#strengths--limitations)
2. [Bidirectional Recurrent Neural Networks (Bi-RNNs)](#2-bidirectional-recurrent-neural-networks-bi-rnns)
   * [Definition](#definition-1)
   * [Why We Need Future Context](#why-we-need-future-context)
   * [Dual Independent Track Architecture](#dual-independent-track-architecture)
   * [Strengths & Limitations](#strengths--limitations-1)
3. [Architectural Synthesis: Deep Bidirectional Networks](#3-architectural-synthesis-deep-bidirectional-networks)
4. [Mathematical & Tensor Flow Mechanics](#4-mathematical--tensor-flow-mechanics)
   * [Vertical Sequence Routing (`return_sequences`)](#vertical-sequence-routing-return_sequences)
   * [Bidirectional Hidden State Fusion](#bidirectional-hidden-state-fusion)
5. [Hands-On Implementations (IMDB Benchmark)](#5-hands-on-implementations-imdb-benchmark)
   * [1. Deep SimpleRNN](#1-deep-simplernn)
   * [2. Deep LSTM](#2-deep-lstm)
   * [3. Deep GRU](#3-deep-gru)
   * [4. Bidirectional SimpleRNN](#4-bidirectional-simplernn)
   * [5. Bidirectional LSTM](#5-bidirectional-lstm)
   * [6. Bidirectional GRU](#6-bidirectional-gru)
   * [7. Deep Bidirectional GRU (with Regularization)](#7-deep-bidirectional-gru-with-regularization)
6. [Comparative Summary & Key Takeaways](#6-comparative-summary--key-takeaways)

---

## 1. Deep Recurrent Neural Networks (Deep RNNs)

### Definition
A **Deep Recurrent Neural Network** (whether composed of SimpleRNN, LSTM, or GRU units) is a sequential architecture where multiple recurrent layers are stacked vertically on top of one another.

```
Layer L    [RNN] ──▶ [RNN] ──▶ [RNN] ──▶ [RNN] ──▶ Final Output / Prediction
             ▲         ▲         ▲         ▲
             │         │         │         │  (Hidden States pass upward)
Layer 2    [RNN] ──▶ [RNN] ──▶ [RNN] ──▶ [RNN]
             ▲         ▲         ▲         ▲
             │         │         │         │  (Layer 1 outputs act as Layer 2 inputs)
Layer 1    [RNN] ──▶ [RNN] ──▶ [RNN] ──▶ [RNN]
             ▲         ▲         ▲         ▲
            x_1       x_2       x_3       x_4  (Raw Input Tokens)
           (t=1)     (t=2)     (t=3)     (t=4)
```

### Why We Need Vertical Depth
Shallow recurrent networks (a single recurrent layer feeding into an output dense layer) can only learn flat, basic surface patterns. In complex data domains, sequential patterns exhibit high hierarchical structure:
* **Natural Language:** Lower layers extract basic morphological and lexical patterns (word stems, parts of speech). Intermediate layers parse syntactic structures (phrase boundaries, clauses). Upper layers capture semantic intent, pragmatic tone, and narrative context.
* **Time-Series / Audio:** Lower layers learn localized frequency variations or raw spikes; higher layers learn seasonal trends or macro-level market regimes.

### The 2D Computational Grid
Deep RNNs introduce vertical spatial depth alongside the temporal horizontal axis:
1. **Horizontal Processing (Temporal Axis):** Information flows through time steps $t_1, t_2, \dots, t_T$ via internal recurrent weight connections $W_{hh}$.
2. **Vertical Processing (Depth Axis):** At each time step $t$, the hidden activation vector $h_t^{(l-1)}$ emitted by layer $l-1$ serves as the input vector $x_t^{(l)}$ for layer $l$.

### Strengths & Limitations
* **Strengths:**
  * Drastically expanded hypothesis capacity for intricate modeling.
  * Captures multi-level hierarchical representations automatically.
* **Limitations:**
  * **Worsened Gradient Decay:** Backpropagating through both time ($T$ steps) and depth ($L$ layers) exacerbates vanishing or exploding gradients (necessitating LSTMs, GRUs, or residual connections).
  * **Inherent Computational Bottleneck:** Recurrent dependencies prevent parallel execution across the temporal axis, and vertical stacking further prevents layer $l$ from finishing until layer $l-1$ computes its representations.

---

## 2. Bidirectional Recurrent Neural Networks (Bi-RNNs)

### Definition
A **Bidirectional Recurrent Neural Network** processes an input sequence simultaneously in two opposite directions:
1. **Forward Pass:** From past to future ($t_1 \to t_T$).
2. **Backward Pass:** From future to past ($t_T \to t_1$).

```
                 Output Head / Dense Layer
                   ▲         ▲         ▲
                  ( )       ( )       ( )  (Concatenation: [ \vec{h}_t ; \overleftarrow{h}_t ])
                 ┌─┴─────────┼─────────┴─┐
Backward Layer  [RNN] ◀─── [RNN] ◀─── [RNN]  (Reads future to past)
Forward Layer   [RNN] ───▶ [RNN] ───▶ [RNN]  (Reads past to future)
                 ▲           ▲           ▲
                x_1         x_2         x_3  (Input Sequence)
```

### Why We Need Future Context
In a standard causal recurrent network, the hidden state at step $t$ ($h_t$) only retains information from steps $1$ to $t-1$. However, in many tasks, the true meaning of an element depends heavily on words or data points that appear **after** it:

> Example:
> 1. *"He ate an **apple** because he was hungry."* $\to$ Fruit.
> 2. *"He worked at **Apple** for ten years."* $\to$ Technology company.

Reading left-to-right, the word *"Apple"* cannot be resolved with certainty until the subsequent context words (*"because"* vs. *"for ten years"*) are encountered. Bidirectional networks solve this ambiguity by reading from both sides simultaneously.

### Dual Independent Track Architecture
* The Forward layer ($\vec{h}_t$) and the Backward layer ($\overleftarrow{h}_t$) maintain **completely independent weight matrices**.
* They do not share information during their respective sequential sweeps.
* At each time step $t$, the model combines both states (most commonly via concatenation) to produce a combined representation:

$$
y_t = g\left(W_y [\vec{h}_t \,;\, \overleftarrow{h}_t] + b_y\right)
$$

### Strengths & Limitations
* **Strengths:**
  * Delivers complete contextual awareness of the past, present, and future at every single step.
  * Significantly outperforms unidirectional networks on offline NLP tasks (e.g., Sentiment Analysis, Named Entity Recognition, Machine Translation).
* **Limitations:**
  * **Cannot be used for real-time online streaming:** Tasks like real-time stock trading, real-time speech translation, or streaming sensory prediction cannot use Bi-RNNs because future tokens are not yet available.
  * **Double Parameter & Computational Cost:** Requires two complete sets of recurrent weights and two matrix multiplication passes per step.

---

## 3. Architectural Synthesis: Deep Bidirectional Networks

When maximum representational power and complete context are both required, **Deep Bidirectional RNNs** stack multiple bidirectional layers on top of each other:

```
Layer 2 (Bi-RNN)   Forward:   [RNN] ──▶ [RNN] ──▶ [RNN]
                   Backward:  [RNN] ◀── [RNN] ◀── [RNN]
                                ▲         ▲         ▲
                                └─────────┼─────────┘  (Layer 1 outputs feed Layer 2)
Layer 1 (Bi-RNN)   Forward:   [RNN] ──▶ [RNN] ──▶ [RNN]
                   Backward:  [RNN] ◀── [RNN] ◀── [RNN]
                                ▲         ▲         ▲
                               x_1       x_2       x_3
```

Each intermediate Bidirectional layer outputs a 3D sequence tensor where each step contains concatenated forward and backward features, serving as the input sequence to the next bidirectional layer.

---

## 4. Mathematical & Tensor Flow Mechanics

### Vertical Sequence Routing (`return_sequences`)

In Keras/TensorFlow, managing tensor shapes across recurrent depth is governed by `return_sequences`:

| Configuration | Output Tensor Shape | Usage |
| :--- | :--- | :--- |
| `return_sequences=True` | `(batch_size, time_steps, units)` | **Required** for all intermediate recurrent layers so the next stacked layer receives a 3D sequence. |
| `return_sequences=False` | `(batch_size, units)` | Used by the **final** recurrent layer to compress the entire sequence into a single 2D vector for the `Dense` classifier. |

### Bidirectional Hidden State Fusion

When wrapping a layer of size $U$ inside `Bidirectional()`, the default merge mode is concatenation:

$$\vec{h}_t \in \mathbb{R}^U, \quad \overleftarrow{h}_t \in \mathbb{R}^U \implies h_t^{\text{merged}} = [\vec{h}_t \,;\, \overleftarrow{h}_t] \in \mathbb{R}^{2U}$$

* If a layer has `units = 8`, the output dimension of `Bidirectional(layer)` is **16**.
* If `return_sequences=True`, output shape is `(None, 100, 16)`.
* If `return_sequences=False`, output shape is `(None, 16)`.

---

## 5. Hands-On Implementations (IMDB Benchmark)

The following models use the IMDB Sentiment Classification dataset:
* Vocabulary: Top $10,000$ words
* Fixed Sequence Length (`MAX_LEN`): $100$ tokens (Pre-padded)
* Latent Embedding Dimension: $32$
* Recurrent Hidden Units: $8$

```python
import tensorflow as tf
from keras import Sequential
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional

# Preprocessing
num_words = 10000
MAX_LEN = 100
EMBEDDING_DIM = 32
UNITS = 8

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='pre')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='pre')
```

---

### 1. Deep SimpleRNN

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    SimpleRNN(UNITS, return_sequences=True),   # Layer 1: Emits 3D sequence (None, 100, 8)
    SimpleRNN(UNITS, return_sequences=False),  # Layer 2: Emits terminal 2D state (None, 8)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Breakdown:**
  * `SimpleRNN 1`: $(32 \times 8) + (8 \times 8) + 8 = \mathbf{328}$
  * `SimpleRNN 2`: $(8 \times 8) + (8 \times 8) + 8 = \mathbf{136}$

---

### 2. Deep LSTM

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    LSTM(UNITS, return_sequences=True),
    LSTM(UNITS, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Breakdown:**
  * `LSTM 1`: $4 \times [(32 \times 8) + (8 \times 8) + 8] = 4 \times 328 = \mathbf{1,312}$
  * `LSTM 2`: $4 \times [(8 \times 8) + (8 \times 8) + 8] = 4 \times 136 = \mathbf{544}$

---

### 3. Deep GRU

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    GRU(UNITS, return_sequences=True),
    GRU(UNITS, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Breakdown:**
  * `GRU 1`: $3 \times [(32 \times 8) + (8 \times 8) + (2 \times 8)] = \mathbf{1,008}$ *(includes bias variants)*
  * `GRU 2`: $3 \times [(8 \times 8) + (8 \times 8) + (2 \times 8)] = \mathbf{432}$

---

### 4. Bidirectional SimpleRNN

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    Bidirectional(SimpleRNN(UNITS)),  # Output Shape: (None, 16)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Count:** $2 \times 328 = \mathbf{656}$

---

### 5. Bidirectional LSTM

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(UNITS)),  # Output Shape: (None, 16)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Count:** $2 \times 1,312 = \mathbf{2,624}$

---

### 6. Bidirectional GRU

```python
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    Bidirectional(GRU(UNITS)),  # Output Shape: (None, 16)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```
* **Parameter Count:** $2 \times 1,008 = \mathbf{2,016}$

---

### 7. Deep Bidirectional GRU (with Regularization)

To prevent overfitting across multiple deep bidirectional layers, add `recurrent_dropout` and kernel regularization:

```python
from keras.regularizers import l1_l2

model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM),
    Bidirectional(GRU(UNITS, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))),
    Bidirectional(GRU(UNITS, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))),
    Bidirectional(GRU(UNITS, return_sequences=False)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 6. Comparative Summary & Key Takeaways

| Architecture | Information Processing Flow | Context Visibility | Main Use Case | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Standard RNN / LSTM / GRU** | 1D: Horizontal ($t_1 \to t_T$) | Past only | Online streaming & casual sequence classification | Cannot extract abstract hierarchical semantics |
| **Deep (Stacked) RNN** | 2D: Horizontal ($t$) + Vertical ($l$) | Past only | Complex language understanding, generative modeling | Vanishing gradients across vertical depth; slow training |
| **Bidirectional RNN** | 1D dual track: ($t_1 \to t_T$) & ($t_T \to t_1$) | Full (Past + Future) | Offline NLP, Named Entity Recognition, Translation | Cannot run on live streaming data; $2\times$ parameters |
| **Deep Bidirectional RNN** | 2D dual track across multiple layers | Full (Past + Future) | High-performance NLP representation models | High computational overhead; prone to overfitting without regularizers |

### Essential Rules of Thumb:
1. **Always use `return_sequences=True`** on every stacked recurrent layer except the final one connected to your classification/regression head.
2. **Use Bidirectional layers for offline analysis** where the full document or sound wave is known ahead of time.
3. **Avoid Bidirectional layers for causal forecasting** (such as stock prediction or next-token generation) to avoid lookahead bias and data leakage from the future.