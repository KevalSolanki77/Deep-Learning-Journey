# Day 25: Recurrent Neural Networks (RNNs)

A comprehensive guide and code reference covering the theory, tensor geometry, end-to-end text processing workflow, and hands-on implementations of Recurrent Neural Networks (RNNs) with TensorFlow and Keras.

---

## Table of Contents
1. [The Core Concept: Sequential Data](#1-the-core-concept-sequential-data)
2. [RNN Fundamentals & The 3D Data Tensor](#2-rnn-fundamentals--the-3d-data-tensor)
3. [The 4-Step Text Processing Workflow](#3-the-4-step-text-processing-workflow)
4. [Structural Types of RNNs](#4-structural-types-of-rnns)
5. [Hands-On Code Implementation](#5-hands-on-code-implementation)
   - [Synthetic Workflow: String to Prediction](#a-synthetic-pipeline-from-raw-strings)
   - [Real-World Benchmark: IMDB Sentiment Analysis](#b-real-world-benchmark-imdb-movie-reviews)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)

---

## 1. The Core Concept: Sequential Data

### Definition
**Sequential data** is any data where the order of data points carries critical semantic meaning. Individual elements cannot be fully interpreted in isolation; they only reveal their true context when evaluated in relation to preceding and succeeding elements.

- **"Dog bites man"** vs. **"Man bites dog"**: Both contain identical tokens, but inverted sequence alters the reality completely.

### Real-World Examples
* **Natural Language / Text:** Sentences, dialogue, paragraphs.
* **Time-Series Data:** Stock market indices, weather forecasts, sensor telemetry.
* **Audio & Video:** Acoustic signals over milliseconds, video frames over seconds.

### Why Traditional Neural Networks Fail
* **MLPs (Multi-Layer Perceptrons):** Assume all feature inputs $X_1, X_2, \dots, X_n$ are mutually independent. They cannot maintain temporal memory and demand a fixed, rigid input vector size.
* **Standard CNNs:** Excellent for spatial locality (pixels), but standard convolutions lack dynamic parameter persistence across variable-length time dimensions without specialized modifications (like causal convolutions).

---

## 2. RNN Fundamentals & The 3D Data Tensor

### The Hidden State ($H_t$)
The hidden state acts as the network's recurrent internal memory. At every discrete time step $t$, the hidden state $H_t$ is computed using both the current time step's input $X_t$ and the previous time step's memory $H_{t-1}$:

$$H_t = \tanh(W_{xh} X_t + W_{hh} H_{t-1} + b_h)$$

Where:
* $W_{xh}$: Weight matrix for input-to-hidden transitions.
* $W_{hh}$: Recurrent weight matrix for hidden-to-hidden transitions.
* $b_h$: Bias vector.

### Parameter Sharing
Unlike a dense network that allocates unique weights per feature index, an RNN applies the **exact same weight matrices ($W_{xh}, W_{hh}, W_{hy}$)** across every single time step. This design dramatically minimizes parameter overhead and enables the network to generalize patterns regardless of position in the sequence.

### The 3D Input Tensor Shape
To account for time, data entering an RNN layer must adhere strictly to a 3-dimensional tensor layout:

$$\mathbf{(Batch\ Size,\ Sequence\ Length,\ Feature\ Dimension)}$$

| Dimension | Meaning | Example (Text) |
| :--- | :--- | :--- |
| **Batch Size** | Number of independent samples processed in parallel | $32$ samples |
| **Sequence Length** | Number of steps/words per sequence | $100$ words |
| **Feature Dimension** | Representation size of each individual token | $16$-dimensional embedding |

### The Core Bottleneck: Vanishing & Exploding Gradients
When backpropagating errors across extensive time steps (**Backpropagation Through Time, or BPTT**), gradients repeatedly multiply with $W_{hh}$. 
* If eigenvalues of $W_{hh} < 1$, gradients exponentially shrink to zero ($\to 0$), erasing the model's ability to retain long-term dependencies.
* If eigenvalues $> 1$, gradients explode ($\to \infty$), causing numerical instability.

*(This fundamental limitation motivates modern gated architectures such as LSTMs and GRUs).*

---

## 3. The 4-Step Text Processing Workflow

To transform unformatted, variable strings of text into numerical predictions, data flows through four distinct transformation phases:

```
[ Raw Text Input ]
        │
        ▼
┌───────────────────────────────┐
│ 1. Vectorization Layer        │ ──▶ Tokenizes, lowercases, maps words to integer IDs
└───────────────────────────────┘     Output: 2D Tensor (Batch Size, Sequence Length)
        │
        ▼
┌───────────────────────────────┐
│ 2. Embedding Layer            │ ──▶ Maps discrete IDs to continuous vector space
└───────────────────────────────┘     Output: 3D Tensor (Batch Size, Sequence Length, Embedding Dim)
        │
        ▼
┌───────────────────────────────┐
│ 3. Recurrent Layer (RNN)      │ ──▶ Unrolls over time, updates internal state H_t
└───────────────────────────────┘     Output: 2D Tensor (Batch Size, RNN Units) [if return_sequences=False]
        │
        ▼
┌───────────────────────────────┐
│ 4. Output Head (Dense)        │ ──▶ Computes probability/class via Sigmoid / Softmax
└───────────────────────────────┘
```

1. **Vectorization:** Cleans text, standardizes casing, removes punctuation, breaks strings into discrete tokens, and indexes tokens as integers.
2. **Embedding:** Serves as a trainable lookup dictionary mapping discrete integer IDs to continuous, dense geometric vectors where semantically related tokens reside close to one another in vector space.
3. **RNN Layer:** Sequentially parses tokens step-by-step, propagating hidden state representations across time.
4. **Output Head:** A standard `Dense` classification layer that condenses the terminal hidden state $H_T$ into target classes (e.g., using `sigmoid` for binary classification).

---

## 4. Structural Types of RNNs

RNN configurations dynamically adapt based on sequence dimensionality requirements:

| Architecture | Input Shape | Output Shape | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **One-to-One** | Single step ($T_x = 1$) | Single step ($T_y = 1$) | Traditional Feedforward / Vanilla Classification |
| **One-to-Many** | Single step ($T_x = 1$) | Sequence ($T_y > 1$) | Image Captioning (Image $\to$ Sentence) |
| **Many-to-One** | Sequence ($T_x > 1$) | Single step ($T_y = 1$) | Sentiment Analysis, Document Classification |
| **Many-to-Many (Synced)** | Sequence ($T_x = N$) | Sequence ($T_y = N$) | Frame-by-frame Video Action Tagging, POS Tagging |
| **Many-to-Many (Unsynced)** | Sequence ($T_x = N$) | Sequence ($T_y = M$) | Machine Translation, Text Summarization (Seq2Seq) |

---

## 5. Hands-On Code Implementation

### A. Synthetic Pipeline: From Raw Strings

```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Hyperparameters
VOCAB_SIZE = 1000       # Vocabulary capacity
MAX_LEN = 8             # Fixed sequence length (padding/truncating target)
EMBEDDING_DIM = 16      # Latent vector dimension per token
RNN_UNITS = 32          # Latent dimension of hidden state vector H_t

# 1. Corpus
texts = [
    "keval solanki",
    "i love rnns",
    "i am learning deep learning",
    "deep learning is fun",
]

# 2. Text Vectorization Layer
vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize="lower_and_strip_punctuation",
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(texts)

# 3. Model Architecture
model = Sequential([
    layers.Input(shape=(1,), dtype="string"),
    vectorizer,
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    layers.SimpleRNN(units=RNN_UNITS, activation="relu", return_sequences=False),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
```

#### Layer Parameter Breakdown
* **Embedding Layer:** $\text{Vocab Size} \times \text{Embedding Dim} = 1000 \times 16 = \mathbf{16,000}$
* **SimpleRNN Layer:**
  $$\text{Params} = (\text{Embedding Dim} \times \text{Units}) + (\text{Units} \times \text{Units}) + \text{Units (bias)}$$
  $$\text{Params} = (16 \times 32) + (32 \times 32) + 32 = 512 + 1024 + 32 = \mathbf{1,568}$$
* **Dense Layer:** $(32 \times 1) + 1 = \mathbf{33}$
* **Total Parameters:** $\mathbf{17,601}$

---

### B. Real-World Benchmark: IMDB Movie Reviews

Demonstration of binary sentiment classification using pre-tokenized sequences from the IMDB benchmark dataset.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import layers, Sequential

# Load dataset (top 10,000 vocabulary words)
MAX_FEATURES = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 16
RNN_UNITS = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

# Uniform sequence padding (Post-padding)
X_train = pad_sequences(X_train, padding="post", maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, padding="post", maxlen=MAX_LENGTH)

# Model Assembly
imdb_model = Sequential([
    layers.Input(shape=(MAX_LENGTH,)),
    layers.Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM),
    layers.SimpleRNN(units=RNN_UNITS, activation="relu", return_sequences=False),
    layers.Dense(1, activation="sigmoid")
])

imdb_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

imdb_model.summary()

# Training
history = imdb_model.fit(
    X_train, 
    y_train, 
    epochs=5, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)
```

---

## 6. Summary & Key Takeaways

1. **Order Encodes Meaning:** Sequential models are required whenever element ordering determines semantic outcome.
2. **Short-Term Memory ($H_t$):** Simple RNNs compress contextual history through hidden recurrent vectors, sharing weights across all steps.
3. **Tensor Geometry:** Input data to any Keras RNN layer must be shaped as `(Batch_Size, Sequence_Length, Features)`.
4. **The Vanishing Gradient Trap:** Vanilla RNNs struggle with long sequence horizons ($>50\text{--}100$ steps), establishing the architectural requirement for gated units (**LSTMs** and **GRUs**).