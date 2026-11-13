# Day 31: Luong Attention (Multiplicative Attention)

A comprehensive theoretical and practical guide examining the computational and architectural bottlenecks of **Bahdanau (Additive) Attention**, how **Luong (Multiplicative) Attention** streamlines sequence alignment, its mathematical scoring functions, and a complete English-to-Hindi Neural Machine Translation implementation in TensorFlow and Keras.

## Table of Contents

1. [Problems with Bahdanau Attention](#1-problems-with-bahdanau-attention)
2. [How Luong Attention Solves Each Problem](#2-how-luong-attention-solves-each-problem)
3. [How Luong Attention Calculates Attention](#3-how-luong-attention-calculates-attention)
   * [Step-by-Step Pipeline](#step-by-step-pipeline)
   * [Luong's Three Alignment Score Functions](#luongs-three-alignment-score-functions)
   * [Global vs. Local Attention](#global-vs-local-attention)
4. [Architectural Comparison: Bahdanau vs. Luong](#4-architectural-comparison-bahdanau-vs-luong)
5. [Hands-On Implementation (English-to-Hindi NMT)](#5-hands-on-implementation-english-to-hindi-nmt)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)

---

## 1. Problems with Bahdanau Attention

While Bahdanau et al. (2014) successfully eliminated the single-vector information bottleneck in Seq2Seq models, their **Additive Attention** architecture introduced several computational and structural inefficiencies:

1. **Expensive Feedforward Alignment Calculation:**
   Bahdanau attention computes alignment scores by passing the concatenated or summed query and key through a separate single-layer neural network with a $\tanh$ activation and three learnable parameter matrices ($W_a, U_a, v_a$):
   $$e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$
   Evaluating this non-linear transformation for every $(i, j)$ pair across long sequences creates significant matrix multiplication overhead and slows down training.

2. **Awkward Step Ordering (Reliance on Previous Decoder State $s_{i-1}$):**
   In Bahdanau's formulation, attention is computed *before* the current decoder RNN step executes, using the **previous** decoder hidden state ($s_{i-1}$). The resulting context vector $c_i$ is then concatenated with the word embedding and fed *into* the decoder LSTM cell. This couples the attention mechanism tightly inside the recurrent cell loop, making layer stacking and modular implementation cumbersome.

3. **Limited Scalability to Long Sequences:**
   Because Bahdanau attention always attends globally across every single source token using additive neural projections, memory bandwidth and compute time scale poorly as document lengths grow.

---

## 2. How Luong Attention Solves Each Problem

Introduced by Minh-Thang Luong, Hieu Pham, and Christopher D. Manning (2015) in *"Effective Approaches to Attention-based Neural Machine Translation"*, **Luong Attention** simplifies and accelerates the mechanism:

| Problem in Bahdanau Attention | Luong Attention Solution |
| :--- | :--- |
| **Slow Feedforward Scoring ($W_a, U_a, v_a$)** | Replaces additive neural networks with **fast matrix dot products** ($h_t^\top \bar{h}_s$), leveraging highly optimized GPU BLAS operations with zero extra parameters for basic dot-product scoring. |
| **Pre-RNN Coupling using $s_{i-1}$** | Computes attention **after** the decoder RNN step executes, using the **current** top-layer decoder hidden state ($h_t$). This creates a clean, modular pipeline: `Encoder -> Decoder -> Attention -> Output`. |
| **Uniform Global Bottleneck** | Introduces both **Global Attention** (attending to all source words) and **Local Attention** (focusing on a differentiable window of size $2D + 1$ around a predicted aligned position $p_t$). |

---

## 3. How Luong Attention Calculates Attention

### Step-by-Step Pipeline

At each decoding time step $t$:

![Loung Attention Calculation](https://i.ibb.co/TMNkdWwg/luong-attention.webp)

1. **Run Decoder RNN First:**
   Compute the current decoder hidden state $h_t$ from the previous hidden state $h_{t-1}$ and current input embedding.
2. **Compute Alignment Scores:**
   Compare the current decoder state $h_t$ with every encoder hidden state $\bar{h}_s$:
   $$e_{ts} = \operatorname{score}(h_t, \bar{h}_s)$$
3. **Normalize via Softmax:**
   Convert raw alignment scores into the attention weight vector $a_t$:
   $$a_t(s) = \frac{\exp(\operatorname{score}(h_t, \bar{h}_s))}{\sum_{s'=1}^{T_x} \exp(\operatorname{score}(h_t, \bar{h}_{s'}))}$$
4. **Compute Context Vector ($c_t$):**
   Take the weighted average of all encoder hidden states:
   $$c_t = \sum_{s=1}^{T_x} a_t(s)\bar{h}_s$$
5. **Compute Attentional Hidden State ($\tilde{h}_t$):**
   Concatenate the context vector $c_t$ and current decoder state $h_t$, then pass them through a `tanh` dense projection layer:
   $$\tilde{h}_t = \tanh\left(W_c [c_t \,;\, h_t]\right)$$
6. **Predict Target Word:**
   Pass $\tilde{h}_t$ through the final `Dense` softmax layer:
   $$P(y_t \mid y_{<t}, x) = \operatorname{softmax}(W_s \tilde{h}_t)$$

### Luong's Three Alignment Score Functions

Luong et al. proposed three ways to compute $\operatorname{score}(h_t, \bar{h}_s)$:

1. **Dot (Pure Multiplicative):**
   $$\operatorname{score}(h_t, \bar{h}_s) = h_t^\top \bar{h}_s$$
   *Fastest and parameter-free; requires encoder and decoder hidden dimensions to match.*

2. **General (Bilinear):**
   $$\operatorname{score}(h_t, \bar{h}_s) = h_t^\top W_a \bar{h}_s$$
   *Introduces a single learnable weight matrix $W_a$, allowing encoder and decoder dimensions to differ.*

3. **Concat (Additive variant):**
   $$\operatorname{score}(h_t, \bar{h}_s) = v_a^\top \tanh(W_a [h_t \,;\, \bar{h}_s])$$
   *Concatenates both states into a single projection matrix $W_a$.*

### Global vs. Local Attention

* **Global Attention:** Considers all encoder hidden states $\bar{h}_1 \dots \bar{h}_{T_x}$ when deriving $c_t$ (implemented by `tf.keras.layers.Attention()`).
* **Local Attention:** Predicts a single aligned source position $p_t \in [1, T_x]$ and applies a Gaussian-weighted window $[p_t - D, p_t + D]$ around it, reducing computational cost on very long sequences.

---

## 4. Architectural Comparison: Bahdanau vs. Luong

| Feature | Bahdanau Attention (2014) | Luong Attention (2015) |
| :--- | :--- | :--- |
| **Alignment Mechanism** | Additive (Feedforward NN with $\tanh$) | Multiplicative (Dot Product / Bilinear) |
| **Decoder Query State** | Previous step state ($s_{i-1}$) | Current step state ($h_t$) |
| **Execution Order** | Attention $\to$ Decoder RNN $\to$ Output | Decoder RNN $\to$ Attention $\to$ $\tilde{h}_t$ $\to$ Output |
| **Attentional Layer** | Concatenated directly before Softmax | Passes $[c_t; h_t]$ through $\tilde{h}_t = \tanh(W_c [c_t; h_t])$ |
| **Keras Layer** | `tf.keras.layers.AdditiveAttention()` | `tf.keras.layers.Attention()` |
| **Computational Speed** | Slower (extra dense projections per pair) | Faster (optimized matrix dot-product multiplication) |

---

## 5. Hands-On Implementation (English-to-Hindi NMT)

[Notebook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-31-Luong-Attention/Luong-Attention.ipynb)

---

## 6. Summary & Key Takeaways

1. **Dot-Product Efficiency:** Luong Attention replaces Bahdanau's additive neural network alignment with fast dot products ($h_t^\top \bar{h}_s$), reducing parameter count in the alignment layer to zero.
2. **Cleaner Post-RNN Flow:** By querying the encoder with the *current* decoder hidden state $h_t$ rather than $s_{i-1}$, attention cleanly wraps around the output of any recurrent decoder stack.
3. **The Attentional Hidden State ($\tilde{h}_t$):** Concatenating `[c_t; h_t]` and projecting through $\tanh(W_c [c_t; h_t])$ fuses source alignment context with target recurrent state before vocabulary classification.
4. **Foundation for Transformers:** Luong's dot-product similarity scoring directly inspired Scaled Dot-Product Attention in the Transformer architecture (`Attention Is All You Need`).