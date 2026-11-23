# Day 35: Layer Normalization & The Complete Transformer Encoder Block

Layer Normalization (LayerNorm) standardizes the hidden features inside a single token or time step on its own, without aggregating statistics across a mini-batch. In Transformers and RNNs, sequential inputs vary in length and change across time steps. LayerNorm stabilizes activations and gradients across every step, and it completes the set of components needed to build a full Transformer Encoder block.

## Table of Contents

1. [What Layer Normalization Does](#1-what-layer-normalization-does)
2. [The Three-Step LayerNorm Computation](#2-the-three-step-layernorm-computation)
   * [Normalization Axis: BatchNorm vs. LayerNorm](#normalization-axis-batchnorm-vs-layernorm)
   * [Mathematical Equations](#mathematical-equations)
   * [Role of Learnable Scale and Shift](#role-of-learnable-scale-and-shift)
3. [Why Batch Normalization Breaks in RNNs and Transformers](#3-why-batch-normalization-breaks-in-rnns-and-transformers)
4. [Residual Connections and LayerNorm Placement](#4-residual-connections-and-layernorm-placement)
5. [Assembling the Complete Transformer Encoder Block](#5-assembling-the-complete-transformer-encoder-block)
   * [Data Flow Through the Encoder](#data-flow-through-the-encoder)
   * [How Each Component Contributes](#how-each-component-contributes)
6. [Summary](#6-summary)

## 1. What Layer Normalization Does

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton introduced Layer Normalization in 2016 to stabilize training in recurrent and feedforward networks.

As weights update during training, the distribution of activations entering deeper layers shifts. This internal covariate shift causes erratic gradients and slows convergence. Convolutional Neural Networks (CNNs) fix this issue with Batch Normalization (BatchNorm), which computes mean and variance across a mini-batch of images for each channel.

Sequential models operate on 3D tensors of shape $(B, T, d)$, where $B$ is batch size, $T$ is sequence length, and $d$ is the hidden feature dimension ($d_{\text{model}}$). LayerNorm computes statistics across the $d$ hidden features of each individual token vector. Every token normalizes on its own, independent of batch size, sequence length, padding tokens, or surrounding time steps.

## 2. The Three-Step LayerNorm Computation

### Normalization Axis: BatchNorm vs. LayerNorm

* **Batch Normalization:** Computes one mean and variance per feature index across all $B$ samples and $T$ positions in the mini-batch.
* **Layer Normalization:** Computes one mean and variance per token vector $x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ across its $d$ hidden features.

### Mathematical Equations

For any single token or time-step vector $x \in \mathbb{R}^d$:

1. **Compute Mean and Variance:**
   Calculate the scalar mean ($\mu$) and variance ($\sigma^2$) across the $d$ features of that single vector:

   $$
   \mu = \frac{1}{d} \sum_{i=1}^{d} x_i
   $$

   $$
   \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
   $$

2. **Standardize to Zero Mean and Unit Variance:**
   Subtract the mean and divide by the standard deviation. A small constant $\epsilon \approx 10^{-5}$ prevents division by zero:

   $$
   \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$

3. **Apply Learnable Scale and Shift:**
   Multiply element-wise by a learned gain vector $\gamma \in \mathbb{R}^d$ and add a learned bias vector $\beta \in \mathbb{R}^d$:

   $$
   y_i = \gamma_i \hat{x}_i + \beta_i
   $$

### Role of Learnable Scale and Shift

Forcing every activation vector to have zero mean and unit variance limits the expressiveness of non-linear layers such as ReLU and GELU. Initializing $\gamma$ to $1$ and $\beta$ to $0$ allows backpropagation to learn the optimal scale and offset for each feature dimension. If an unnormalized representation works best for a specific layer, the network can learn to invert the standardization.

## 3. Why Batch Normalization Breaks in RNNs and Transformers

| Limitation | Impact on RNNs | Impact on Transformers |
| :--- | :--- | :--- |
| **Variable Sequence Lengths** | Shorter sequences need zero-padding. At later time steps, batch statistics average over fewer real tokens, producing noisy gradients. | Padding tokens distort batch-wide means and variances, and attention masks complicate cross-batch averages. |
| **Time-Step Instability** | Recurrent weights multiply repeatedly across steps, causing activation magnitudes to drift over time. Storing per-step running averages fails when test sequences exceed training lengths. | Token statistics vary by grammatical role and position. A single batch average misses these positional shifts. |
| **Batch Size Dependency** | Backpropagation through time consumes heavy memory, forcing small mini-batches where BatchNorm estimates become erratic. | Large Language Models often train or fine-tune with micro-batches of 1 or 2 sequences per GPU, where cross-batch variance breaks down. |
| **Train vs. Inference Mismatch** | During step-by-step generation, BatchNorm relies on frozen training averages that diverge from live recurrent states. | Autoregressive decoding generates one token at a time. LayerNorm computes exact statistics on the fly, keeping training and inference math identical. |

## 4. Residual Connections and LayerNorm Placement

Transformers pair every LayerNorm layer with a residual skip connection. Adding the sub-layer input $x$ to the sub-layer output $\text{Sublayer}(x)$ creates an unobstructed path for gradients to flow from the final loss back to the input embeddings.

Two placement conventions exist in practice:

1. **Post-LN (Original 2017 Transformer):**
   LayerNorm sits outside the residual branch, normalizing after the addition:

   $$
   x_{\text{out}} = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)
   $$

   Post-LN produces larger gradients near the output layers and requires a learning-rate warm-up schedule at the start of training.

2. **Pre-LN (Modern LLMs such as GPT-2, GPT-3, and LLaMA):**
   LayerNorm sits inside the residual branch, normalizing inputs before they enter the sub-layer:

   $$
   x_{\text{out}} = x + \text{Sublayer}\big(\text{LayerNorm}(x)\big)
   $$

   Pre-LN leaves the main residual path unscaled from input to output, stabilizing gradient flow in deep networks with dozens of layers.

## 5. Assembling the Complete Transformer Encoder Block

With Layer Normalization covered, we have all the building blocks required for a complete Transformer Encoder block.

### Data Flow Through the Encoder

1. **Raw Tokens** $\rightarrow$ **Token Embeddings** (dense vectors of size $d_{\text{model}}$)
2. **Add Positional Encoding** (injects token position signals from Day 34)
3. **Enter Transformer Encoder Block:**
   * **Sub-Layer 1:** Multi-Head Self-Attention (Days 32 and 33)
   * **Add & Norm 1:** Residual skip connection + Layer Normalization (Day 35)
   * **Sub-Layer 2:** Position-Wise Feed-Forward Network (FFN)
   * **Add & Norm 2:** Residual skip connection + Layer Normalization (Day 35)
4. **Output:** Contextualized token representations of shape $(T, d_{\text{model}})$, stacked $N$ layers deep (for example, $N = 6$ in the base Transformer or $N = 12$ in BERT-Base).

### How Each Component Contributes

* **Token Embeddings + Positional Encoding (Day 34):** Maps words to dense vectors and adds sinusoidal or learned position signals so parallel self-attention layers track word order.
* **Multi-Head Self-Attention (Days 32 and 33):** Projects embeddings into $h$ parallel Query, Key, and Value subspaces. Each head captures distinct syntactic, semantic, or positional relationships across all tokens at once, then concatenates and projects the results back to $d_{\text{model}}$.
* **Add & Norm 1 (Day 35):** Adds the pre-attention input back to the attention output via a residual connection and standardizes each token vector with LayerNorm.
* **Position-Wise Feed-Forward Network (FFN):** Applies two linear transformations with a non-linear activation (ReLU or GELU) independently to each token vector. The first layer typically expands the dimension from $d_{\text{model}}$ to $4 \cdot d_{\text{model}}$, and the second projects it back to $d_{\text{model}}$.
* **Add & Norm 2 (Day 35):** Applies a second residual addition and LayerNorm, outputting refined contextual representations ready for the next encoder layer or a downstream task head.

## 6. Summary

* **Per-Token Normalization:** LayerNorm computes mean and variance across the $d$ hidden features of each individual token, removing dependence on batch size and sequence padding.
* **Consistent Execution:** LayerNorm uses the exact same per-token formula during training and autoregressive inference, avoiding frozen running statistics.
* **Complete Encoder Blueprint:** Combining Positional Encoding, Multi-Head Self-Attention, Position-Wise Feed-Forward Networks, residual connections, and LayerNorm forms a complete Transformer Encoder block.