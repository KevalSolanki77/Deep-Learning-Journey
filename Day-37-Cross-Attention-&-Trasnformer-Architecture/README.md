# Day 37: Cross-Attention & The Complete Transformer Architecture

Cross-Attention connects two separate data sequences inside a Transformer, allowing a target sequence to query and extract relevant context from a source sequence. Combined with Masked Multi-Head Attention, Positional Encoding, and Layer Normalization, Cross-Attention completes the full Encoder-Decoder Transformer architecture.

## Table of Contents

1. [What is Cross-Attention?](#1-what-is-cross-attention)
2. [How Cross-Attention Works](#2-how-cross-attention-works)
   * [Splitting Origins of Q, K, and V](#splitting-origins-of-q-k-and-v)
   * [Matching, Weighting, and Equations](#matching-weighting-and-equations)
   * [Tensor Dimensions Across Sequences](#tensor-dimensions-across-sequences)
3. [Cross-Attention vs. Self-Attention](#3-cross-attention-vs-self-attention)
4. [Common Use Cases](#4-common-use-cases)
5. [The Complete Transformer Architecture](#5-the-complete-transformer-architecture)
   * [End-to-End Architectural Diagram](#end-to-end-architectural-diagram)
   * [Step-by-Step Signal Flow](#step-by-step-signal-flow)
6. [Summary](#6-summary)

---

## 1. What is Cross-Attention?

**Cross-Attention** (also called **Encoder-Decoder Attention**) is an attention mechanism that enables one sequence of representations to focus on and pull information from a different sequence.

Self-attention operates inside a single sequence to map internal relationships among its own tokens. Cross-attention acts as a bridge between two distinct sequences, such as an English source sentence and a Hindi target translation, or a text prompt and a grid of image patches.

---

## 2. How Cross-Attention Works

Cross-attention uses the three core projection matrices, Queries ($Q$), Keys ($K$), and Values ($V$), and splits their sources across two separate representations:

### Splitting Origins of Q, K, and V

* **Queries ($Q$) come from the Target Sequence:** In an Encoder-Decoder model, the decoder generates $Q$ from its current hidden representations (what it has generated so far and what information it needs next).
* **Keys ($K$) and Values ($V$) come from the Source Sequence:** The encoder processes the entire original input sequence and provides $K$ (the searchable index of source features) and $V$ (the semantic content of those source features).

### Matching, Weighting, and Equations

1. **Compute Alignment Scores:** Multiply the target's Query matrix ($Q_{\text{dec}}$) by the transposed source Key matrix ($K_{\text{enc}}^\top$) to measure how relevant every source token is to every target position.
2. **Scale and Normalize:** Divide by $\sqrt{d_k}$ and apply row-wise softmax to obtain attention weights that sum to $1$ across the source sequence length.
3. **Extract Context:** Multiply the normalized weights by the source Value matrix ($V_{\text{enc}}$) to produce a weighted sum of source representations for each target step:

$$
\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^\top}{\sqrt{d_k}}\right) V_{\text{enc}}
$$

In Multi-Head Cross-Attention, this operation runs across $h$ parallel heads, concatenates the outputs, and projects them through $W^O$.

### Tensor Dimensions Across Sequences

Because the source sequence length ($n_{\text{src}}$) and target sequence length ($n_{\text{tgt}}$) often differ, the attention weight matrix is rectangular rather than square:

| Matrix | Shape | Origin |
| :--- | :--- | :--- |
| **Queries ($Q_{\text{dec}}$)** | $(n_{\text{tgt}}, d_k)$ | Projected from Decoder hidden states |
| **Keys ($K_{\text{enc}}$)** | $(n_{\text{src}}, d_k)$ | Projected from Encoder final outputs |
| **Values ($V_{\text{enc}}$)** | $(n_{\text{src}}, d_v)$ | Projected from Encoder final outputs |
| **Attention Weight Matrix** | $(n_{\text{tgt}}, n_{\text{src}})$ | Maps each target position to all source positions |
| **Cross-Attention Output** | $(n_{\text{tgt}}, d_v)$ | Source context aligned to target sequence positions |

---

## 3. Cross-Attention vs. Self-Attention

| Feature | Cross-Attention | Self-Attention |
| :--- | :--- | :--- |
| **Input Sources** | Two distinct sequences (e.g., source and target) | One single sequence |
| **Origin of Q, K, V** | $Q$ from sequence A; $K$ and $V$ from sequence B | $Q$, $K$, and $V$ all come from sequence A |
| **Masking** | Unmasked over source tokens (decoder can view the entire source sequence at once, ignoring only padding tokens) | Masked in decoders to block future tokens; unmasked in encoders |
| **Attention Matrix Shape** | Rectangular: $(n_{\text{tgt}} \times n_{\text{src}})$ | Square: $(n \times n)$ |
| **Primary Goal** | Align and transfer context from one sequence to another | Model internal relationships within a single sequence |

---

## 4. Common Use Cases

* **Machine Translation:** As a decoder generates a translated sentence token by token, cross-attention lets each target position look back at the original source sentence to focus on the exact words being translated, handling differences in word order across languages.
* **Multimodal AI (Text-to-Image and Image Captioning):** Cross-attention connects different data modalities. In text-to-image diffusion models, noisy image patches act as Queries ($Q$) while text prompt embeddings act as Keys ($K$) and Values ($V$), guiding visual synthesis to match the text description.
* **Automatic Speech Recognition (Audio Transcription):** Models align generated text tokens ($Q$) with processed acoustic spectrogram frames ($K, V$) to transcribe spoken audio into written words.

---

## 5. The Complete Transformer Architecture

Combining Cross-Attention with the components from Days 32 through 36 yields the full Encoder-Decoder Transformer introduced by Vaswani et al. (2017).

### End-to-End Architectural Diagram

![Transformer Architecture](https://i.ibb.co/0jnC9D85/Screenshot-2026-11-25-151537.png)
image from Research Paper "Attention is all you need"
### Step-by-Step Signal Flow

1. **Input Processing (Encoder Side):**
   * Source tokens pass through an Embedding layer and receive Positional Encodings (Day 34).
   * Each Encoder block runs unmasked **Multi-Head Self-Attention** (Days 32 and 33) followed by a **Position-Wise Feed-Forward Network (FFN)**, wrapped with residual skip connections and **Layer Normalization** (Day 35).
   * After $N$ stacked encoder layers, the final encoder output serves as the $K$ and $V$ memory bank for the decoder.
2. **Target Processing (Decoder Side):**
   * Shifted target tokens pass through an Embedding layer and receive Positional Encodings.
   * **Sub-Layer 1 (Masked Multi-Head Self-Attention):** Each target token attends strictly to itself and earlier target tokens using a causal $-\infty$ mask (Day 36), followed by Add & Norm.
   * **Sub-Layer 2 (Multi-Head Cross-Attention):** The decoder uses its masked self-attention output as Queries ($Q$) to attend over the encoder's Keys ($K$) and Values ($V$), pulling relevant source context into every target step, followed by Add & Norm.
   * **Sub-Layer 3 (Position-Wise FFN):** Refines each token representation independently through two dense layers and a non-linear activation, followed by Add & Norm.
3. **Output Generation Head:**
   * After $N$ stacked decoder layers, a final **Linear layer** projects the hidden vectors to the target vocabulary size, and **Softmax** converts the logits into next-token probabilities.

---

## 6. Summary

* **Two-Sequence Bridge:** Cross-Attention takes Queries ($Q$) from the target sequence and Keys ($K$) and Values ($V$) from the source sequence, producing a rectangular $(n_{\text{tgt}} \times n_{\text{src}})$ alignment matrix.
* **Unmasked Source Access:** Unlike decoder self-attention, cross-attention leaves all non-padded source tokens visible so the decoder can inspect any part of the input at any generation step.
* **Full Transformer Synthesis:** Stacking Multi-Head Self-Attention, Positional Encoding, Layer Normalization, Masked Self-Attention, Cross-Attention, and Feed-Forward Networks forms the complete Encoder-Decoder Transformer architecture.