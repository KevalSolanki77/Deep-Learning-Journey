# Day 27: Gated Recurrent Units (GRUs)

A comprehensive guide and reference examining why Traditional RNNs and LSTMs required an alternative, the architectural innovations introduced by the Gated Recurrent Unit (GRU), the mathematical foundations of its dual-gate design, and a direct trade-off analysis.

---

## Table of Contents
1. [Why Traditional RNNs and LSTMs Needed an Alternative](#1-why-traditional-rnns-and-lstms-needed-an-alternative)
   * [The Problem with Traditional RNNs: "Short-Term Memory"](#the-problem-with-traditional-rnns-short-term-memory)
   * [The Problem with LSTMs: "Over-Engineering"](#the-problem-with-lstms-over-engineering)
2. [The GRU Solution: Streamlining the Architecture](#2-the-gru-solution-streamlining-the-architecture)
   * [Core Architectural Changes](#core-architectural-changes)
   * [Structural Comparison: LSTM vs. GRU](#structural-comparison-lstm-vs-gru)
3. [Architecture of a GRU & Mathematical Equations](#3-architecture-of-a-gru--mathematical-equations)
   * [Step 1: The Reset Gate ($r_t$)](#step-1-the-reset-gate-r_t)
   * [Step 2: The Update Gate ($z_t$)](#step-2-the-update-gate-z_t)
   * [Step 3: Candidate Hidden State ($\tilde{h}_t$)](#step-3-candidate-hidden-state-tildeh_t)
   * [Step 4: Final Hidden State ($h_t$)](#step-4-final-hidden-state-h_t)
4. [Parameter Count Comparison: Mathematical Derivation](#4-parameter-count-comparison-mathematical-derivation)
5. [Summary of Core Trade-offs](#5-summary-of-core-trade-offs)
6. [TensorFlow / Keras Implementation Snippet](#6-tensorflow--keras-implementation-snippet)

---

## 1. Why Traditional RNNs and LSTMs Needed an Alternative

To understand why Kyunghyun Cho et al. introduced the Gated Recurrent Unit in 2014, it is necessary to examine the limitations of what came before:

### The Problem with Traditional RNNs: "Short-Term Memory"
Traditional RNNs process sequential data by passing a single hidden state vector ($h_t$) from one time step to the next:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

* **The Vanishing Gradient Trap:** When training via Backpropagation Through Time (BPTT), the error gradients are propagated backwards across time steps. Because this operation repeatedly multiplies the weight matrices ($W_{hh}$) and activation derivatives:
  $$\prod_{k=1}^T \frac{\partial h_k}{\partial h_{k-1}}$$
  fractions shrink exponentially toward zero ($\to 0$).
* **Consequence:** Early layers receive near-zero weight updates, rendering the network unable to capture dependencies separated by more than $10\text{--}20$ steps.

### The Problem with LSTMs: "Over-Engineering"
LSTMs resolved the vanishing gradient issue by adding a dedicated additive memory highway called the **Cell State ($C_t$)**, managed by three distinct neural network gates:
* Forget Gate ($f_t$)
* Input Gate ($i_t$)
* Output Gate ($o_t$)

While effective, this architecture comes with notable costs:
* **High Parameter Overhead:** Managing four separate affine transformations (three gates + one candidate state) leads to high GPU memory consumption.
* **Slower Convergence:** A large parameter footprint requires more iterations and wall-clock time to train.
* **Prone to Overfitting:** On small-to-medium sized datasets, the high model capacity of LSTMs frequently overfits before generalizing.

---

## 2. The GRU Solution: Streamlining the Architecture

The Gated Recurrent Unit maintains the additive gradient flow and long-term memory capabilities of an LSTM, but streamlines the gating pipeline.

```
LSTM (3 Gates + 1 Cell State) ───▶ Simplified into ───▶ GRU (2 Gates + Combined Hidden State)
```

### Core Architectural Changes

1. **Eliminating the Separate Cell State:**
   Unlike the LSTM (which maintains both a hidden state $h_t$ and a cell state $C_t$), a GRU merges them into a **single unified hidden state ($h_t$)**. This unified state serves as both internal memory and the external recurrent output.

2. **Collapsing the Gate Count from 3 to 2:**
   * **Update Gate ($z_t$):** Replaces both the LSTM Forget and Input gates. It simultaneously decides what past information to carry forward ($1 - z_t$) and what fraction of the candidate memory to write ($z_t$).
   * **Reset Gate ($r_t$):** Decides how much of the past hidden state should be ignored when computing the new candidate representation.

3. **Fewer Parameters:**
   With only two gates plus one candidate state calculation, GRUs utilize **3 weight sets** instead of the LSTM’s **4 weight sets**, resulting in approximately **$25\%$ fewer parameters**.

### Structural Comparison: LSTM vs. GRU

![LSTM vs GRU](https://i.ibb.co/HTcTDhwg/lstm-gru.jpg)
Image from Euron

---

## 3. Architecture of a GRU & Mathematical Equations

At each discrete time step $t$, the GRU takes the current input vector $x_t \in \mathbb{R}^d$ and the previous hidden state $h_{t-1} \in \mathbb{R}^h$.

![GRU Architecture](https://i.ibb.co/GQh1K9Q4/gru.webp)
Image from ResearchGate 

### Step 1: The Reset Gate ($r_t$)
Determines how much of the previous hidden state $h_{t-1}$ to disregard when computing candidate memory. If $r_t \approx 0$, the unit discards past memory and acts as if reading the first symbol of a sequence.

$$
r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)
$$

### Step 2: The Update Gate ($z_t$)
Acts as both the forget and input gates simultaneously. It dictates how much of the previous state $h_{t-1}$ should be preserved versus how much the new candidate state should be adopted.

$$
z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)
$$

### Step 3: Candidate Hidden State ($\tilde{h}_t$)
Computes the proposed new content for the memory vector. Notice how the reset gate $r_t$ modulates the influence of the previous state $h_{t-1}$ using the Hadamard element-wise product ($\odot$):

$$
\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h)
$$

> *Note:* If $r_t = 0$, then $r_t \odot h_{t-1} = 0$, meaning the candidate state is computed purely from the current input $x_t$.

### Step 4: Final Hidden State ($h_t$)
Combines the past state and the candidate state via linear interpolation controlled by the update gate $z_t$:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

* If $z_t \to 0$: The network ignores the new candidate and directly copies past memory $h_{t-1}$ forward through time.
* If $z_t \to 1$: The network overwrites past memory with the newly computed candidate representation $\tilde{h}_t$.

---

## 4. Parameter Count Comparison: Mathematical Derivation

Let:
* $d$ = Dimension of input vector $x_t$
* $h$ = Number of recurrent hidden units

### Vanilla RNN
Only 1 state equation:

$$
\text{Params}_{\text{RNN}} = h \cdot (d + h + 1)
$$

### LSTM
Requires 4 full transformations (Forget, Input, Output, and Candidate):

$$
\text{Params}_{\text{LSTM}} = 4 \times \left[ h \cdot (d + h + 1) \right]
$$

### GRU
Requires 3 full transformations (Reset Gate, Update Gate, and Candidate State):

$$
\text{Params}_{\text{GRU}} = 3 \times \left[ h \cdot (d + h + 1) \right]
$$

### Numeric Example:
Assume an input vector dimension $d = 64$ and hidden units $h = 128$:
* **RNN:** $128 \times (64 + 128 + 1) = 128 \times 193 = \mathbf{24,704}$ params
* **LSTM:** $4 \times [128 \times 193] = \mathbf{98,816}$ params
* **GRU:** $3 \times [128 \times 193] = \mathbf{74,112}$ params ($\mathbf{25\%}$ reduction from LSTM)

---

## 5. Summary of Core Trade-offs

| Feature | Traditional RNN | LSTM | GRU (Streamlined Solution) |
| :--- | :--- | :--- | :--- |
| **Long-Term Memory** | Poor (Suffers from Vanishing Gradients) | Excellent | Excellent (Filters out noise with gates) |
| **Internal States** | 1 State ($h_t$) | 2 States ($h_t$ and $C_t$) | 1 State ($h_t$) |
| **Gating Mechanism** | None | 3 Gates (Input, Forget, Output) | 2 Gates (Update, Reset) |
| **Mathematical Formulae** | 1 Equation | 6 Equations | 4 Equations |
| **Parameter Scale** | $1 \times \text{Weight Sets}$ | $4 \times \text{Weight Sets}$ | $3 \times \text{Weight Sets}$ ($\sim 25\%$ reduction vs. LSTM) |
| **Training Speed** | Fastest per step (inaccurate over long data) | Slower (High parameter footprint) | $\mathbf{15\% \text{ to } 30\%}$ faster than LSTM |
| **Data Efficiency** | Requires least data | Prone to overfitting on small datasets | Performs better on smaller/medium datasets |

---

## 6. TensorFlow / Keras Implementation Snippet

Swapping between LSTM and GRU in Keras involves replacing the recurrent layer call; the tensor input shape $(Batch, Steps, Features)$ remains identical:

```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Sequence classification / forecasting baseline using GRU
model = Sequential([
    layers.Input(shape=(60, 1)),                  # (Sequence length = 60, Features = 1)
    layers.GRU(units=64, return_sequences=True),   # First recurrent layer passes sequences
    layers.Dropout(0.2),
    layers.GRU(units=64, return_sequences=False),  # Last recurrent layer outputs terminal state
    layers.Dropout(0.2),
    layers.Dense(units=1, activation='linear')     # Prediction head
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```