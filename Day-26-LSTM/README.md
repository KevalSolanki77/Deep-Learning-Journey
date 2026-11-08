# Day 26: Long Short-Term Memory Networks (LSTMs)

A comprehensive guide and reference covering the transition from vanilla RNNs to LSTMs, the mechanics of gating mechanisms, formal mathematical derivations, and two real-world implementations using TensorFlow and Keras: **Stock Market Forecasting** and an **NLP Next-Word Text Generator**.

---

## Table of Contents
1. [The Problem with Vanilla RNNs](#1-the-problem-with-vanilla-rnns)
2. [The Core Idea of LSTMs](#2-the-core-idea-of-lstms)
3. [How LSTMs Differ from RNNs](#3-how-lstms-differ-from-rnns)
4. [Architecture of an LSTM Cell & Mathematical Equations](#4-architecture-of-an-lstm-cell--mathematical-equations)
   - [Step 1: The Forget Gate ($f_t$)](#step-1-the-forget-gate-f_t)
   - [Step 2: The Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$)](#step-2-the-input-gate-i_t--candidate-memory-tildec_t)
   - [Step 3: Updating the Cell State ($C_t$)](#step-3-updating-the-cell-state-c_t)
   - [Step 4: The Output Gate ($o_t$) & Hidden State ($H_t$)](#step-4-the-output-gate-o_t--hidden-state-h_t)
5. [Hands-On Model 1: Nifty 50 Stock Market Predictor](#5-hands-on-model-1-nifty-50-stock-market-predictor)
6. [Hands-On Model 2: Next-Word Text Generator](#6-hands-on-model-2-next-word-text-generator)
7. [Summary & Key Takeaways](#7-summary--key-takeaways)

---

## 1. The Problem with Vanilla RNNs

In standard Recurrent Neural Networks (RNNs), the hidden state update is computed through continuous matrix multiplications and non-linear squashing activations:

$$
H_t = \tanh(W_{xh} X_t + W_{hh} H_{t-1} + b_h)
$$

When computing gradients across long sequences via **Backpropagation Through Time (BPTT)**, the gradient of the loss at time step $T$ with respect to the initial hidden state $H_0$ involves a chain product of Jacobian matrices:

$$
\frac{\partial L}{\partial H_0} = \frac{\partial L}{\partial H_T} \prod_{j=1}^{T} \frac{\partial H_j}{\partial H_{j-1}} = \frac{\partial L}{\partial H_T} \prod_{j=1}^{T} \operatorname{diag}(1 - H_j^2) W_{hh}^T
$$

### The Consequences:
1. **Vanishing Gradient Problem:** Because the derivative of $\tanh$ is bounded in the interval $(0, 1]$, and weights in $W_{hh}$ are often small, repeatedly multiplying these terms over tens or hundreds of steps causes the gradient to decay exponentially towards zero ($\to 0$). The network loses the ability to learn long-term dependencies.
2. **Exploding Gradient Problem:** If the eigenvalues of $W_{hh}$ are greater than $1$, gradients grow exponentially ($\to \infty$), causing numerical overflow (`NaN` loss) and destabilized weight updates.

---

## 2. The Core Idea of LSTMs

Introduced by **Sepp Hochreiter and Jürgen Schmidhuber (1997)**, the **Long Short-Term Memory (LSTM)** network solves the vanishing gradient problem by introducing a dedicated communication highway: the **Cell State ($C_t$)**.

- **Additive Gradient Flow:** Unlike the hidden state in standard RNNs, which undergoes multiplicative transformation via $W_{hh}$ at every step, the cell state runs down the sequence with linear interactions (mostly addition).
- **Constant Error Carousel (CEC):** Because changes to the cell state are additive, errors can backpropagate through hundreds of time steps without exponentially decaying to zero.
- **Dynamic Gates:** Information is explicitly regulated via specialized "gates"—small feedforward layers with sigmoid activations ($\sigma$) that output values between $0$ (completely block) and $1$ (completely allow).

---

## 3. How LSTMs Differ from RNNs

| Feature | Standard (Vanilla) RNN | Long Short-Term Memory (LSTM) |
| :--- | :--- | :--- |
| **Internal Memory States** | Single state: Hidden state ($H_t$). | Dual states: Hidden state ($H_t$) & Cell state ($C_t$). |
| **Information Transfer** | Overwritten and transformed at every step via matrix multiplications. | Preserved through a linear additive highway ($C_t$), modified only via gates. |
| **Gating Mechanisms** | None. | Three distinct gates: Forget Gate, Input Gate, Output Gate. |
| **Parameter Count** | $\text{Units} \times (\text{Input\_Dim} + \text{Units} + 1)$ | $4 \times \left[ \text{Units} \times (\text{Input\_Dim} + \text{Units} + 1) \right]$ |
| **Effective Sequence Length** | Short ($< 10\text{--}50$ steps before context loss). | Long ($100\text{--}1000+$ steps). |

---

## 4. Architecture of an LSTM Cell & Mathematical Equations

At each discrete time step $t$, an LSTM cell receives three pieces of information:
1. Current input vector: $X_t \in \mathbb{R}^{d}$
2. Previous hidden state (short-term memory): $H_{t-1} \in \mathbb{R}^{h}$
3. Previous cell state (long-term memory): $C_{t-1} \in \mathbb{R}^{h}$

!['LSTM Architecture'](https://i.ibb.co/d0GNXhq4/Screenshot-2026-11-08-170933.png)
Image from GeeksforGeeks

### Step 1: The Forget Gate ($f_t$)
Determines what percentage of past memory should be discarded from the cell state. A value of $0$ means "completely drop this memory," and $1$ means "retain this memory entirely."

$$
f_t = \sigma(W_f \cdot [H_{t-1}, X_t] + b_f)
$$

### Step 2: The Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$)
Decides what new information should be written into the cell state:
- **Input Gate ($i_t$):** Regulates *which* dimensions to update (scale between $0$ and $1$).
- **Candidate State ($\tilde{C}_t$):** Creates the new candidate values squashed between $[-1, 1]$.

$$
i_t = \sigma(W_i \cdot [H_{t-1}, X_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_c \cdot [H_{t-1}, X_t] + b_c)
$$

### Step 3: Updating the Cell State ($C_t$)
Combines the filtered past memory with the scaled candidate information using element-wise addition ($\odot$ represents the Hadamard element-wise product):

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

*Because this operation is purely additive, gradients can pass through time step after time step without diminishing.*

### Step 4: The Output Gate ($o_t$) & Hidden State ($H_t$)
Decides what information from the updated cell state should be emitted as the new hidden state $H_t$ (which acts as the short-term context and the output to subsequent network layers):

$$
o_t = \sigma(W_o \cdot [H_{t-1}, X_t] + b_o)
$$

$$
H_t = o_t \odot \tanh(C_t)
$$

---

## 5. Hands-On Model 1: Nifty 50 Stock Market Predictor

A stacked LSTM architecture built to forecast the closing price of the **Nifty 50 Index (`^NSEI`)** based on a historical **60-day sliding window**.

```
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Fetch Historic Stock Data
df = yf.download('^NSEI', start='2016-01-01', end='2026-11-01')
data = df[['Close']]

# 2. Train-Test Split (Splitting by Date)
train_raw = data[data.index.year < 2026].values
test_raw = data[data.index.year >= 2026].values

# 3. Min-Max Normalization (0 to 1 scale)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

# 4. Generate 60-Day Sliding Windows
TIME_STEPS = 60
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(TIME_STEPS, len(train_scaled)):
    X_train.append(train_scaled[i - TIME_STEPS:i, 0])
    y_train.append(train_scaled[i, 0])

for i in range(TIME_STEPS, len(test_scaled)):
    X_test.append(test_scaled[i - TIME_STEPS:i, 0])
    y_test.append(test_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape into 3D Tensor: [Batch, Time Steps, Features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 5. Build Stacked LSTM Regressor
stock_model = Sequential([
    layers.Input(shape=(TIME_STEPS, 1)),
    layers.LSTM(units=64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(units=64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(units=1, activation='linear')
])

stock_model.compile(optimizer='adam', loss='mean_squared_error')
stock_model.summary()

# 6. Model Training
stock_model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.1
)
```

### Parameter Calculation (Layer 1):
- Input Dimension ($d$) = $1$
- Output Hidden Units ($h$) = $64$
- Each gate has weights for $X$ ($d \times h$), weights for $H$ ($h \times h$), and biases ($h$).
- Formula: $4 \times [h \times (d + h + 1)] = 4 \times [64 \times (1 + 64 + 1)] = 4 \times [64 \times 66] = \mathbf{16,896}$

---

## 6. Hands-On Model 2: Next-Word Text Generator

An autoregressive language model trained on Montgomery's novel *The Blue Castle* using integer tokenization, dense embeddings, high-capacity recurrent layers, and categorical cross-entropy.

```
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Read & Preprocess Text File
with open("/content/blue_castle.txt", "r", encoding="utf8") as file:
    raw_lines = file.readlines()

clean_text = " ".join(raw_lines)
clean_text = clean_text.replace('\n', ' ').replace('\r', '').replace('\ufeff', '')
words = clean_text.lower().split()

# 2. Text Vectorization
SEQUENCE_LEN = 10
vectorizer = layers.TextVectorization(
    max_tokens=10000,
    standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(words)
VOCAB_SIZE = len(vectorizer.get_vocabulary())

# Convert raw word tokens to 1D integer stream
token_stream = vectorizer(words).numpy().flatten()

# 3. Create Sliding Context Windows (N-Gram Inputs -> Next Token Target)
X_seq, y_target = [], []
for i in range(SEQUENCE_LEN, len(token_stream)):
    X_seq.append(token_stream[i - SEQUENCE_LEN:i])
    y_target.append(token_stream[i])

X_seq = np.array(X_seq)
y_target = np.array(y_target)

# 4. Neural Network Architecture
EMBEDDING_DIM = 64
LSTM_UNITS = 1024

nlp_model = Sequential([
    layers.Input(shape=(SEQUENCE_LEN,), dtype='int32'),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    layers.LSTM(units=LSTM_UNITS, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(units=LSTM_UNITS, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(units=VOCAB_SIZE, activation='softmax')
])

# Memory-efficient sparse loss avoids one-hot encoded targets
nlp_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
nlp_model.summary()

# 5. Training
nlp_model.fit(
    X_seq,
    y_target,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# 6. Text Generation Loop
def generate_text(seed_phrase, next_words=10):
    vocab = vectorizer.get_vocabulary()
    generated_output = seed_phrase

    for _ in range(next_words):
        tokens = vectorizer([generated_output]).numpy().flatten()
        
        # Enforce exact sequence length window
        if len(tokens) > SEQUENCE_LEN:
            tokens = tokens[-SEQUENCE_LEN:]
        elif len(tokens) < SEQUENCE_LEN:
            tokens = np.pad(tokens, (SEQUENCE_LEN - len(tokens), 0), 'constant')

        tokens = np.reshape(tokens, (1, SEQUENCE_LEN))
        
        # Predict probability distribution
        pred = nlp_model.predict(tokens, verbose=0)
        best_id = np.argmax(pred, axis=-1)[0]
        word = vocab[best_id]

        if word in ["", "[UNK]"]:
            break

        generated_output += " " + word

    return generated_output

# Demonstration
print(generate_text("the blue castle was", next_words=5))
```

---

## 7. Summary & Key Takeaways

1. **The Core Fix:** Vanilla RNNs fail over long time spans due to continuous squashing multiplications leading to vanishing gradients. LSTMs fix this by providing an additive **cell state conveyor belt**.
2. **Three Specialized Gates:**
   - **Forget Gate ($f_t$):** Decides what stale historical details to clear from memory.
   - **Input Gate ($i_t$ & $\tilde{C}_t$):** Calculates new updates to store in memory.
   - **Output Gate ($o_t$):** Selects which aspects of the memory should appear in the immediate hidden state.
3. **Sequence Passing in Keras:**
   - When stacking multiple recurrent layers, intermediate layers **must** set `return_sequences=True` so the next recurrent unit receives the 3D sequence tensor `(batch, time_steps, features)`.
   - The final recurrent layer uses `return_sequences=False` (default) when connected to a flat `Dense` prediction head.