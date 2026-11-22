# Day 34: Positional Encoding

A comprehensive conceptual guide to **Positional Encoding**—the mechanism that restores sequential order to Transformer architectures. This document explores why self-attention is inherently blind to word order, how positional signals are injected into token embeddings, the trigonometric mathematics behind sinusoidal encodings, and the evolution of positional methods from the original Transformer to modern Large Language Models.

## Table of Contents

1. [What is Positional Encoding?](#1-what-is-positional-encoding)

2. [The Core Problem & Solution](#2-the-core-problem--solution)

   * [The Problem: Permutation Invariance](#the-problem-permutation-invariance)

   * [The Solution: Additive Positional Injection](#the-solution-additive-positional-injection)

   * [Why Addition Instead of Concatenation?](#why-addition-instead-of-concatenation)

3. [Sinusoidal Positional Encoding (Original Transformer)](#3-sinusoidal-positional-encoding-original-transformer)

   * [Mathematical Formulation](#mathematical-formulation)

   * [Property 1: Multi-Frequency Waves (The Binary Clock Analogy)](#property-1-multi-frequency-waves-the-binary-clock-analogy)

   * [Property 2: Linear Relative Distance via Rotation](#property-2-linear-relative-distance-via-rotation)

4. [Evolution of Positional Encoding Methods](#4-evolution-of-positional-encoding-methods)

   * [1. Sinusoidal Encoding (Fixed / Deterministic)](#1-sinusoidal-encoding-fixed--deterministic)

   * [2. Learned Positional Embeddings (BERT, GPT-2)](#2-learned-positional-embeddings-bert-gpt-2)

   * [3. Rotary Position Embedding — RoPE (LLaMA, Mistral, PaLM)](#3-rotary-position-embedding--rope-llama-mistral-palm)

   * [4. Attention with Linear Biases — ALiBi (BLOOM, MPT)](#4-attention-with-linear-biases--alibi-bloom-mpt)

5. [Comparative Summary of Methods](#5-comparative-summary-of-methods)

6. [Summary & Key Takeaways](#6-summary--key-takeaways)

## 1. What is Positional Encoding?

**Positional Encoding** is a mathematical technique that injects information about the relative or absolute position of tokens into a Transformer's input representations. Because Transformers abandon recurrence and process all words in a sequence in parallel, positional encoding provides the model with its sole awareness of word order and sequence structure.

## 2. The Core Problem & Solution

### The Problem: Permutation Invariance

In older recurrent models (RNNs, LSTMs, GRUs), word order was built directly into the computation: token $t_1$ had to be processed before $t_2$, which had to be processed before $t_3$.

Transformers eliminate recurrence in favor of **Self-Attention**, which computes pairwise dot products between all tokens simultaneously:

* Consider the two sentences:
  1. *"The **dog** chased the **cat**."*
  2. *"The **cat** chased the **dog**."*

* Without positional information, self-attention treats a sentence as an **unordered set (bag of vectors)**. Swapping the positions of *"dog"* and *"cat"* produces the exact same pairwise attention scores and identical output representations for each word. This property is called **permutation invariance**.

### The Solution: Additive Positional Injection

To break permutation invariance without reintroducing slow sequential loops, Vaswani et al. (2017) proposed creating a dedicated **Positional Encoding vector** ($PE$) for every position index $\text{pos} \in \{0, 1, 2, \dots, n-1\}$ with the exact same dimensionality ($d_{\text{model}}$) as the word embeddings.

Before entering the first Transformer layer, the positional vector is added element-wise to the semantic token embedding:

$$
X_{\text{input}}(\text{pos}) = \text{Embedding}(\text{token}_{\text{pos}}) + PE(\text{pos})
$$

```
Token:           "The"            "dog"           "chased"
                   │                │                │
                   ▼                ▼                ▼
Semantic Emb:   [e_0]            [e_1]            [e_2]     (What the word means)
                   +                +                +
Positional Enc: [PE_0]           [PE_1]           [PE_2]    (Where the word sits)
                   │                │                │
                   ▼                ▼                ▼
Final Input:    [z_0]            [z_1]            [z_2]     ──▶ Enters Self-Attention

```

### Why Addition Instead of Concatenation?

A natural question is why Transformers **add** the positional vector to the word embedding rather than **concatenating** them side-by-side:

1. **Preserves Dimensionality:** Concatenating two $512$-dimensional vectors doubles the model width to $1024$, increasing parameter counts and FLOPs across every subsequent projection matrix ($W^Q, W^K, W^V, W^O$) and feed-forward layer.

2. **High-Dimensional Orthogonality:** In high-dimensional vector spaces (e.g., $d_{\text{model}} = 512$ or $4096$), two independently structured vectors occupy nearly orthogonal subspaces. Through training, the linear projection layers ($W^Q, W^K, W^V$) effortlessly learn to separate and read both semantic meaning and positional coordinates from the summed vector without interference.

## 3. Sinusoidal Positional Encoding (Original Transformer)

Rather than simply assigning raw integers ($0, 1, 2, \dots$—which grow arbitrarily large and destabilize gradients) or normalized fractions ($0.0$ to $1.0$—which change step sizes depending on total sentence length), the original Transformer uses **interleaved sine and cosine waves** of varying frequencies.

### Mathematical Formulation

For a token at position $\text{pos}$ and embedding dimension index $2i$ (even) or $2i + 1$ (odd):

$$
PE_{(\text{pos},\, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE_{(\text{pos},\, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)
$$

Where:

* $\text{pos}$: The token's index in the sequence ($0, 1, 2, \dots$).

* $i$: The dimension pair index ($0 \le i < d_{\text{model}} / 2$).

* $10000$: A large base wavelength constant chosen so wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

### Property 1: Multi-Frequency Waves (The Binary Clock Analogy)

As the dimension index $i$ increases from $0$ to $d_{\text{model}} / 2$, the denominator $10000^{2i / d_{\text{model}}}$ grows exponentially, causing the wave frequency to decrease smoothly:

* **Low Dimensions ($i \approx 0$, High Frequency):** The sine and cosine waves oscillate rapidly, completing a full cycle every few tokens. These fast waves allow the model to sharply distinguish immediately adjacent words (e.g., position $4$ vs. position $5$).

* **High Dimensions ($i \approx d_{\text{model}}/2$, Low Frequency):** The waves oscillate very slowly across thousands of positions. These slow waves provide a stable, macro-level coordinate for long-range paragraph or document positioning.

> **Analogy:** This mirrors how a **binary counter** works. In binary numbers ($000, 001, 010, 011, 100 \dots$), the rightmost bit flips on every single step (high frequency), the middle bit flips every two steps, and the leftmost bit flips slowly over long intervals (low frequency). Sinusoidal encoding is the smooth, continuous floating-point equivalent of a binary clock.

### Property 2: Linear Relative Distance via Rotation

In natural language, **relative distance** (e.g., *"two words to the left"*) often matters much more than absolute index (e.g., *"index 47"*).

Because of the trigonometric angle-addition identities:

$$
\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)
$$

$$
\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)
$$

For any fixed offset $k$, the positional vector $PE_{\text{pos} + k}$ can be expressed as a **fixed linear transformation (a 2D rotation matrix)** of $PE_{\text{pos}}$ that depends **only on the offset $k$**, completely independent of the starting position $\text{pos}$:

$$
\begin{bmatrix} PE_{(\text{pos}+k,\, 2i)} \\ PE_{(\text{pos}+k,\, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(\text{pos},\, 2i)} \\ PE_{(\text{pos},\, 2i+1)} \end{bmatrix}
$$

This mathematical property allows self-attention weight matrices to easily learn consistent relative-distance filters (such as *"attend to the word 1 step behind me"*) regardless of where a phrase appears in a document.

## 4. Evolution of Positional Encoding Methods

As language models scaled from $512$ tokens to hundreds of thousands of tokens, researchers developed improved ways to encode position:

### 1. Sinusoidal Encoding (Fixed / Deterministic)

* **Used in:** Original Transformer (2017).

* **How it works:** Pre-computes deterministic sine/cosine waves and adds them to input embeddings.

* **Strengths:** Zero learnable parameters; deterministic; theoretically capable of extrapolating to sequence lengths longer than those seen during training.

* **Limitations:** Additive injection at the input layer gradually dilutes as signals pass through dozens of deep Transformer blocks; dot products between shifted sinusoidal vectors are not strictly monotonic with distance.

### 2. Learned Positional Embeddings (BERT, GPT-2)

* **Used in:** BERT, RoBERTa, GPT-2, ViT (Vision Transformer).

* **How it works:** Treats position indices ($0, 1, \dots, L_{\text{max}}-1$) just like vocabulary words in a trainable embedding lookup table, allowing backpropagation to learn optimal position vectors directly from data.

* **Strengths:** Highly flexible and adapts naturally to specific data modalities (such as 2D image patches in Vision Transformers).

* **Limitations:** Strictly bounded by a hard maximum sequence length ($L_{\text{max}}$, such as $512$ in BERT or $1024$ in GPT-2). The model has no embedding vector for position $L_{\text{max}} + 1$ and cannot extrapolate to longer contexts without architectural surgery.

### 3. Rotary Position Embedding — RoPE (LLaMA, Mistral, PaLM)

* **Used in:** LLaMA 1/2/3, Mistral, Gemma, Qwen, PaLM.

* **How it works:** Instead of adding positional vectors to input embeddings before the network, RoPE injects position **directly inside every self-attention layer** by rotating the Query ($Q$) and Key ($K$) vectors in 2D subspaces by an angle proportional to their position ($\text{pos} \cdot \theta_i$).

* **Strengths:** When computing the dot product $Q_m^\top K_n$, the absolute angles cancel out and leave a function that depends **purely on the relative distance $(m - n)$**. It preserves vector norms, decays smoothly with distance, and supports context-window extension techniques (like YaRN and NTK-aware scaling).

### 4. Attention with Linear Biases — ALiBi (BLOOM, MPT)

* **Used in:** BLOOM, MPT, Falcon.

* **How it works:** Adds no positional vectors to embeddings or $Q/K$ projections at all. Instead, it subtracts a static, non-learned linear penalty proportional to token distance directly from the raw attention scores ($Q K^\top$) before the softmax layer:

$$
\text{Score}(i, j) = q_i^\top k_j - m \cdot |i - j|
$$

* **Strengths:** Zero extra parameters; extremely fast; naturally penalizes distant tokens while excelling at zero-shot length extrapolation (e.g., trained on $2\text{k}$ tokens and evaluated cleanly at $16\text{k}+$ tokens).

## 5. Comparative Summary of Methods

| Method | Injection Point | Relative vs. Absolute | Learnable Parameters | Length Extrapolation | Notable Models | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Sinusoidal** | Added to input embeddings | Absolute (with linear relative shift property) | None (Fixed math formula) | Moderate | Original Transformer | 
| **Learned Embeddings** | Added to input embeddings | Strictly Absolute | Yes ($L_{\text{max}} \times d_{\text{model}}$) | Poor (Hard cap at $L_{\text{max}}$) | BERT, GPT-2, ViT | 
| **RoPE (Rotary)** | Rotates $Q$ and $K$ inside attention | Relative via rotational inner product | None (Fixed rotation angles) | Strong (with frequency scaling) | LLaMA, Mistral, Gemma | 
| **ALiBi** | Bias subtracted from $Q K^\top$ scores | Strictly Relative distance penalty | None (Fixed head slopes $m$) | Excellent | BLOOM, MPT | 

## 6. Summary & Key Takeaways

1. **Restoring Word Order:** Self-attention is permutation-invariant by design; positional encoding breaks this symmetry so Transformers can distinguish *"dog chased cat"* from *"cat chased dog."*

2. **Multi-Scale Sinusoidal Waves:** Alternating sine and cosine functions across embedding dimensions act like a continuous binary clock—high-frequency waves capture local token order, while low-frequency waves capture global document position.

3. **Relative Distance via Rotation:** Trigonometric identities guarantee that any positional shift $k$ corresponds to a constant linear rotation matrix, making it easy for attention heads to learn relative token offsets.

4. **Modern Shift to Attention-Level Position:** While early Transformers added positional vectors once at the input layer, modern LLMs inject relative positional geometry directly into the attention mechanism (via **RoPE** or **ALiBi**) to achieve superior long-context scaling.