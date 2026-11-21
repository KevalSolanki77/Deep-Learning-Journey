# Day 33: Multi-Head Attention

A comprehensive conceptual guide to **Multi-Head Attention** the architectural enhancement that enables Transformer models to analyze language and sequences from multiple perspectives simultaneously. This document explains why single-head self-attention suffers from information dilution, walks through the four-step mathematical pipeline of multi-head projection and fusion, proves why parallel heads incur no extra computational penalty, and compares single-head versus multi-head architectures.

## Table of Contents

1. [What is Multi-Head Attention?](#1-what-is-multi-head-attention)
2. [Why We Need Multi-Head Attention](#2-why-we-need-multi-head-attention)
   * [Preventing Information Dilution](#preventing-information-dilution)
   * [Capturing Multiple Linguistic Perspectives](#capturing-multiple-linguistic-perspectives)
3. [How Multi-Head Attention Works: The Four-Step Pipeline](#3-how-multi-head-attention-works-the-four-step-pipeline)
   * [Visual Architecture Diagram](#visual-architecture-diagram)
   * [Step 1: Linear Projections (Splitting into Subspaces)](#step-1-linear-projections-splitting-into-subspaces)
   * [Step 2: Parallel Scaled Dot-Product Attention](#step-2-parallel-scaled-dot-product-attention)
   * [Step 3: Concatenation](#step-3-concatenation)
   * [Step 4: Final Linear Projection ($W^O$)](#step-4-final-linear-projection-w^o)
   * [Tensor Dimension Flow Table](#tensor-dimension-flow-table)
4. [Computational Efficiency: Why 8 Heads Cost the Same as 1](#4-computational-efficiency-why-8-heads-cost-the-same-as-1)
5. [Single Self-Attention vs. Multi-Head Attention](#5-single-self-attention-vs-multi-head-attention)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)

## 1. What is Multi-Head Attention?

**Multi-Head Attention** is an extension of the Scaled Dot-Product Self-Attention mechanism introduced in *"Attention Is All You Need"* (Vaswani et al., 2017). Instead of executing a single attention function over the full-dimensional token embeddings, the Transformer splits the Queries ($Q$), Keys ($K$), and Values ($V$) into $h$ smaller, independent representation subspaces called **attention heads**.

Each head runs self-attention in parallel on its own slice of the vector space, allowing the network to focus on different types of relationships at different positions simultaneously before combining those insights into one unified output representation.

> **Intuition Analogy:** Think of single self-attention as a single reader trying to simultaneously track a novel's grammar, character relationships, emotional tone, and plot timeline in one pass. Multi-Head Attention is like a **panel of specialists** reading the same page together: one specialist tracks grammar, another resolves pronouns, a third tracks tone, and a fourth tracks local word proximity and afterward, they meet to combine their notes.

## 2. Why We Need Multi-Head Attention

While a single self-attention layer can identify relationships between tokens, relying on a single attention map creates two fundamental bottlenecks:

### Preventing Information Dilution

In single self-attention, each token produces only **one** probability distribution (one row of softmax weights) over the entire sequence, resulting in a single weighted average of all Value vectors.

* When a single word needs to attend to multiple other words for completely different reasons (for example, a verb needing to attend to its subject *and* its direct object *and* an adverb modifying it), a single softmax distribution forces all of those signals to compete and average together.
* Averaging distinct semantic and syntactic signals into a single vector blurs sharp, specific details a phenomenon known as **information dilution**.

### Capturing Multiple Linguistic Perspectives

Natural language and complex sequences are inherently multi-dimensional. By giving the model multiple independent sets of projection matrices ($W_i^Q, W_i^K, W_i^V$), different heads naturally specialize during training without explicit supervision:

* **Syntactic Heads:** Link verbs to their subjects or direct objects across long clauses.
* **Coreference Heads:** Resolve ambiguous pronouns (e.g., linking *"it"* back to *"the animal"* rather than *"the street"*).
* **Positional / Local Heads:** Focus heavily on immediately adjacent tokens ($t -1$ and $t+1$) to capture local modifiers, adjectives, and negations (such as *"not"*).
* **Semantic / Topical Heads:** Group thematically related words across distant parts of a paragraph.

## 3. How Multi-Head Attention Works: The Four-Step Pipeline

Given an input sequence matrix $X$ of sequence length $n$ and model embedding dimension $d_{\text{model}}$ (for example, $d_{\text{model}} = 512$), Multi-Head Attention with $h$ parallel heads (for example, $h = 8$) transforms the data through four sequential stages:

### Visual Architecture Diagram

![Multi Head Attention Structure](https://i.ibb.co/qLxvVC4x/Multi-Head-Attention-Mechanism.webp)
Image from GeeksforGeeks

### Step 1: Linear Projections (Splitting into Subspaces)

Instead of projecting $X$ into one large $512$-dimensional space, each head $i \in \{1, \dots, h\}$ has its own learned projection matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, where:

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

For a standard Transformer with $d_{\text{model}} = 512$ and $h = 8$ heads, each head projects the input into a compact $512 \div 8 = 64$-dimensional subspace:

$$
Q_i = X W_i^Q, \qquad K_i = X W_i^K, \qquad V_i = X W_i^V
$$

Because each head learns unique weight matrices, each head asks a completely different "question" about the sequence.

### Step 2: Parallel Scaled Dot-Product Attention

All $h$ heads execute the standard Scaled Dot-Product Attention formula simultaneously and independently on their $64$-dimensional slices:

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
$$

* Notice that scaling is performed by $\sqrt{d_k} = \sqrt{64} = 8$ (the per-head key dimension), keeping variance normalized inside each head.
* This step yields $h$ distinct $n \times n$ attention maps and $h$ separate output matrices, each of shape $(n, 64)$.

### Step 3: Concatenation

The independent output vectors from all $h$ heads are stitched together side-by-side along the feature dimension:

$$
\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)
$$

* Joining $8$ matrices of shape $(n, 64)$ restores the tensor back to its original full embedding width: $(n, 8 \times 64) = (n, 512)$.

### Step 4: Final Linear Projection ($W^O$)

At this stage, the concatenated vector simply holds $8$ separate $64$-dimensional perspectives stacked next to one another without any cross-head interaction. To synthesize these isolated perspectives into one cohesive representation, the concatenated matrix is multiplied by a final learned output weight matrix $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

This final projection mixes information across all heads and prepares a clean $(n, d_{\text{model}})$ tensor for residual connections, Layer Normalization, and the subsequent Feed-Forward Network.

### Tensor Dimension Flow Table

Assuming sequence length $n = 10$, model dimension $d_{\text{model}} = 512$, and $h = 8$ heads ($d_k = d_v = 64$):

| Stage | Operation | Output Shape | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | Token Embeddings ($X$) | $(10, 512)$ | Initial sequence representation |
| **Step 1** | Per-Head Projections ($Q_i, K_i, V_i$) | $8 \times (10, 64)$ | Maps tokens into 8 specialized subspaces |
| **Step 2a** | Per-Head Alignment ($Q_i K_i^\top / \sqrt{64}$) | $8 \times (10, 10)$ | Produces 8 distinct pairwise attention maps |
| **Step 2b** | Per-Head Weighted Sum ($\alpha_i V_i$) | $8 \times (10, 64)$ | Extracts 8 specialized contextual views |
| **Step 3** | Concatenation across Heads | $(10, 512)$ | Reassembles full embedding dimensionality |
| **Step 4** | Output Projection ($\times W^O$) | $(10, 512)$ | Blends insights across all heads into one representation |

## 4. Computational Efficiency: Why 8 Heads Cost the Same as 1

A common misconception is that running $8$ attention heads requires $8\times$ more computation than running a single attention head. In reality, because the embedding dimension $d_{\text{model}}$ is **divided** evenly across the heads ($d_k = d_{\text{model}} / h$), the total computational complexity remains virtually identical:

* **Single-Head Attention (Full Dimension $d_{\text{model}}$):**
  Computing $Q K^\top$ for sequence length $n$ requires $O(n^2 \cdot d_{\text{model}})$ operations.
* **Multi-Head Attention ($h$ Heads of Dimension $d_{\text{model}} / h$):**
  Each head requires $O(n^2 \cdot \frac{d_{\text{model}}}{h})$ operations. Across all $h$ heads in parallel:

$$
h \times O\left(n^2 \cdot \frac{d_{\text{model}}}{h}\right) = O(n^2 \cdot d_{\text{model}})
$$

Furthermore, on modern GPUs and TPUs, all $h$ heads are batched into a single 4D tensor multiplication, allowing all heads to execute simultaneously in a single hardware kernel call.

## 5. Single Self-Attention vs. Multi-Head Attention

| Feature | Single Self-Attention | Multi-Head Attention |
| :--- | :--- | :--- |
| **Core Mechanism** | Computes one combined attention map across the full embedding space | Splits embeddings into subspaces and runs multiple attention maps in parallel |
| **Linguistic Ability** | Averages all syntactic, semantic, and positional relationships together | Separates grammar, coreference, tone, and local context into specialized heads |
| **Information Fidelity** | Prone to blurring distinct relationships into a single weighted average | Preserves sharp, distinct signals across parallel channels |
| **Cross-Channel Mixing** | None (single weighted sum of Values) | Uses output projection $W^O$ to synthesize insights across all heads |
| **Computational Cost** | $O(n^2 \cdot d_{\text{model}})$ | Virtually identical $O(n^2 \cdot d_{\text{model}})$ due to dimension splitting |

## 6. Summary & Key Takeaways

1. **Eliminating Dilution:** Single self-attention forces every linguistic relationship into one averaged distribution; Multi-Head Attention preserves distinct signals by evaluating multiple subspaces simultaneously.
2. **Four-Step Mechanics:** Project into $h$ smaller $(Q_i, K_i, V_i)$ slices $\to$ compute Scaled Dot-Product Attention in parallel $\to$ concatenate head outputs $\to$ fuse with linear projection $W^O$.
3. **Specialization Without Extra FLOPs:** Splitting $d_{\text{model}}$ into $h$ slices of size $d_{\text{model}} / h$ gives the Transformer rich multi-perspective reasoning at virtually the same computational cost as a single full-width head.