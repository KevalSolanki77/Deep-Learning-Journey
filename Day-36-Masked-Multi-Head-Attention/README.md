# Day 36: Masked Multi-Head Attention

Masked Multi-Head Attention enables a Transformer Decoder to train across entire sequences in parallel while preserving strict left-to-right causal order. By blocking future token positions with negative infinity before the softmax step, it prevents the model from copying upcoming words during training and aligns training behavior with step-by-step text generation.

## Table of Contents

1. [The Famous Line and Core Terms Explained](#1-the-famous-line-and-core-terms-explained)
   * [Autoregressive Execution (Inference Time)](#autoregressive-execution-inference-time)
   * [Non-Autoregressive Execution (Training Time)](#non-autoregressive-execution-training-time)
2. [The Problem with Unmasked Attention and the Causal Solution](#2-the-problem-with-unmasked-attention-and-the-causal-solution)
   * [The Problem: Data Leakage](#the-problem-data-leakage)
   * [The Solution: Masked Multi-Head Attention](#the-solution-masked-multi-head-attention)
3. [Step-by-Step Mathematical Pipeline](#3-step-by-step-mathematical-pipeline)
   * [The Causal Mask Matrix](#the-causal-mask-matrix)
   * [Masked Scaled Dot-Product Equation](#masked-scaled-dot-product-equation)
4. [Key Architectural Properties and Insights](#4-key-architectural-properties-and-insights)
5. [Summary](#5-summary)

## 1. The Famous Line and Core Terms Explained

> **"The Transformer Decoder is autoregressive at inference time and non-autoregressive at training time."**

During training, the Transformer Decoder processes the entire target sequence in parallel using ground-truth text. During actual deployment, it generates output tokens one by one in sequence.

### Autoregressive Execution (Inference Time)

"Auto" means self. When you prompt a language model, the complete answer does not exist yet.

1. The model processes your prompt and predicts one token.
2. It appends that newly generated token to the input sequence.
3. It runs the entire forward pass again to predict the following token.

Because step $t+1$ must wait for step $t$ to finish, inference runs sequentially and requires one forward pass per generated token.

### Non-Autoregressive Execution (Training Time)

During training, the dataset already contains the complete, ground-truth target sentence. Rather than waiting for the model to guess word by word and compound early mistakes, we feed the entire target sequence into the decoder at once using **Teacher Forcing**. The model predicts every next token simultaneously in a single parallel matrix pass, accelerating GPU training by orders of magnitude.

## 2. The Problem with Unmasked Attention and the Causal Solution

### The Problem: Data Leakage

In standard unmasked Multi-Head Attention (used in Transformer Encoders such as BERT), every token attends to every other token in the sequence, including both past and future positions.

If you feed a complete target sentence into a Decoder using unmasked attention, position $i$ can directly inspect position $i+1$ in the input matrix. Instead of learning grammar, semantics, and next-word probabilities, the network copies the ground-truth answer from the future column. When deployed at inference time, where future words do not exist, the model fails.

### The Solution: Masked Multi-Head Attention

To preserve parallel GPU training without leaking future tokens, the decoder applies a **Causal Mask** (also called a look-ahead mask) inside every attention head:

1. **Compute Raw Similarity Scores:** Multiply Queries by transposed Keys ($Q K^\top$) across all token pairs simultaneously.
2. **Apply the Causal Mask ($-\infty$):** Add a mask matrix $M$ that overwrites every upper-right triangle entry (all positions where key index $j >$ query index $i$) with negative infinity ($-\infty$).
3. **Softmax Zero-Out:** Apply the softmax function row by row. Because $e^{-\infty} = 0$, every future token receives an attention weight of $0$. Each position aggregates information strictly from itself and preceding tokens while the entire sequence computes in one parallel pass.

## 3. Step-by-Step Mathematical Pipeline

### The Causal Mask Matrix

For a sequence of length $n = 4$ (for example, `["<start>", "I", "love", "AI"]`), the additive mask matrix $M \in \mathbb{R}^{4 \times 4}$ takes a lower-triangular structure:

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

* **Row 1 (`<start>`):** Attends only to position 1.
* **Row 2 (`I`):** Attends to positions 1 and 2.
* **Row 3 (`love`):** Attends to positions 1, 2, and 3.
* **Row 4 (`AI`):** Attends to positions 1, 2, 3, and 4.

### Masked Scaled Dot-Product Equation

For each attention head $i$, the causal mask $M$ is added directly to the scaled dot-product logits before softmax normalization:

$$
\text{MaskedAttention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\right) V_i
$$

| Step | Operation | Effect on Future Positions ($j > i$) |
| :--- | :--- | :--- |
| **1. Scale** | $\frac{Q_i K_i^\top}{\sqrt{d_k}}$ | Produces raw similarity scores for all pairs |
| **2. Mask** | $+ M$ | Shifts future logits to $-\infty$; leaves past and self logits unchanged ($+0$) |
| **3. Softmax** | $\text{softmax}(\cdot)$ | Evaluates $e^{-\infty} = 0$, assigning zero probability weight to future tokens |
| **4. Aggregate** | $\times V_i$ | Blends Value vectors strictly from past and current positions |

## 4. Key Architectural Properties and Insights

* **The Triangular Visual Signature:** Visualizing a post-softmax causal attention matrix reveals a lower-triangular staircase. Visible past and self connections sit on and below the main diagonal, while blocked future entries sit at zero above the diagonal.
* **Why $-\infty$ Is Required Instead of $0$:** Multiplying or setting future logits to $0$ before softmax fails because $e^0 = 1$, which assigns a positive probability weight to future words. Setting masked entries to $-\infty$ (or a large negative constant such as $-10^9$ in finite precision) drives $e^{-\infty} \to 0$.
* **Behavior During Inference (Prefill vs. Generation):** During autoregressive inference, text generation happens in two phases:
  * **Prefill Phase:** When you submit a multi-word prompt, the decoder processes all prompt tokens in parallel using the causal mask so every prompt token only sees earlier prompt tokens.
  * **Token-by-Token Decoding Phase:** When generating one new token at a time, future tokens do not exist yet, so the newest query vector naturally attends to all cached past keys without needing to block future indices.
* **Head Specialization Within the Past:** Every head shares the same future-blocking mask, yet each head learns separate $W_i^Q, W_i^K, W_i^V$ projections. One head can focus on the immediately preceding word for local syntax, while another looks dozens of tokens back to track subject-verb agreement or topic context.

## 5. Summary

* **Parallel Training, Sequential Inference:** Teacher Forcing allows the decoder to train across all target positions simultaneously, while real-world generation runs one token at a time.
* **Preventing Look-Ahead Leakage:** Adding $-\infty$ to upper-triangular positions before softmax forces future attention weights to $0$, preserving left-to-right causality.
* **Multi-Head Past Context:** Splitting masked attention across parallel heads lets the decoder extract syntactic, positional, and long-range semantic patterns from prior tokens at the same time.