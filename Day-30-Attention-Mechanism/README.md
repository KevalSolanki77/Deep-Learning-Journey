# Day 30: The Attention Mechanism in Deep Learning

A comprehensive theoretical and practical guide exploring why traditional Sequence-to-Sequence models hit a performance wall, how **Bahdanau Additive Attention** dynamically aligns source and target sequences, the mathematical equations powering attention weights, and a complete English-to-Hindi Neural Machine Translation implementation in TensorFlow and Keras.

## Table of Contents

1. [The Problem with Traditional Encoder-Decoder Models](#1-the-problem-with-traditional-encoder-decoder-models)
   * [The Information Bottleneck](#the-information-bottleneck)
   * [Loss of Long-Range Context](#loss-of-long-range-context)
   * [Rigid Summary](#rigid-summary)
2. [The Solution: The Attention Mechanism](#2-the-solution-the-attention-mechanism)
   * [What is Attention?](#what-is-attention)
   * [Core Capabilities](#core-capabilities)
3. [Bahdanau Additive Attention (2014)](#3-bahdanau-additive-attention-2014)
   * [Alignment Score Equation](#alignment-score-equation)
   * [From Raw Scores to Context Vector](#from-raw-scores-to-context-vector)
   * [Visualizing the Attention Pipeline](#visualizing-the-attention-pipeline)
4. [4-Step Keras / TensorFlow Implementation Guide](#4-4-step-keras--tensorflow-implementation-guide)
5. [Complete Hands-On Code (English-to-Hindi NMT)](#5-complete-hands-on-code-english-to-hindi-nmt)
   * [Step 1: Data Cleaning & Vectorization](#step-1-data-cleaning--vectorization)
   * [Step 2: Training Model with AdditiveAttention](#step-2-training-model-with-additiveattention)
   * [Step 3: Standalone Inference Pipeline](#step-3-standalone-inference-pipeline)
6. [Summary & Key Takeaways](#6-summary--key-takeaways)

---

## 1. The Problem with Traditional Encoder-Decoder Models

In a standard Sequence-to-Sequence (Seq2Seq) model without attention, the encoder reads the entire source sentence and discards every intermediate hidden state, retaining only the final state vector to initialize the decoder.

### The Information Bottleneck
Whether the input sentence has 4 words or 60 words, the encoder must compress the entire semantic meaning, syntax, and nuance into a **single, fixed-length context vector** ($h_{T_x}, c_{T_x}$). This creates a severe representational bottleneck.

### Loss of Long-Range Context
As input sequences grow longer, early words get overwritten or diluted by subsequent time steps before the encoder finishes reading the sentence. Consequently, translation quality drops sharply on long, multi-clause sentences.

### Rigid Summary
Every single generated target word is forced to rely on the **exact same static summary vector**, regardless of which specific source word is being translated at that exact moment.

---

## 2. The Solution: The Attention Mechanism

### What is Attention?
**Attention** is a neural network mechanism that creates direct, learnable shortcuts between every decoding output step and **all** encoder input hidden states ($h_1, h_2, \dots, h_{T_x}$), completely eliminating the single-vector bottleneck.

### Core Capabilities

* **Dynamic Focusing:** At each decoding step $i$, the model computes a probability distribution (attention weights summing to $1$) across all source words to decide which input tokens matter most right now.
* **Custom Context Vectors:** Instead of reusing one static summary for the whole sentence, the decoder constructs a fresh, tailored context vector $c_i$ (a weighted sum of all encoder outputs) for **every single predicted word**.
* **Better Cross-Lingual Alignment:** Allows the model to smoothly align source and target words even when languages follow different grammatical word orders—for example, translating English (**Subject–Verb–Object**: *"How are you?"*) into Hindi (**Subject–Object–Verb**: *"आप कैसे हो?"*).

---

## 3. Bahdanau Additive Attention (2014)

Introduced by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio in 2014, **Additive Attention** was the first widely adopted attention mechanism for Neural Machine Translation. Rather than using a simple dot product, it employs a small single-layer feedforward neural network to score how well a source word matches the current decoding step.

### Alignment Score Equation

$$
e_{ij} = v_a^\top \tanh\left(W_a s_{i-1} + U_a h_j\right)
$$

Where:
* **$e_{ij}$**: Raw unnormalized alignment score between target decoding step $i$ and source encoder position $j$.
* **$s_{i-1}$**: Previous hidden state of the decoder (acts as the **Query**).
* **$h_j$**: Hidden state of the encoder at source word $j$ (acts as the **Key / Value**).
* **$W_a, U_a, v_a$**: Learnable weight matrices and alignment vector trained jointly via backpropagation.

### From Raw Scores to Context Vector

1. **Softmax Normalization (Attention Weights):**
   Converts raw alignment scores $e_{ij}$ into normalized attention weights $\alpha_{ij} \in [0, 1]$ that sum to $1$ across all source positions $j = 1 \dots T_x$:
   $$
   \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} = \operatorname{softmax}(e_{ij})
   $$

2. **Weighted Sum (Dynamic Context Vector):**
   Multiplies each encoder hidden state $h_j$ by its attention weight $\alpha_{ij}$ and sums them to produce a custom context vector $c_i$ for target step $i$:
   $$
   c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
   $$

### Visualizing the Attention Pipeline

![Bahdanau Additive Attention Pipeline](https://i.ibb.co/SDDsDtQq/attention.png)

---

## 4. 4-Step Keras / TensorFlow Implementation Guide

Adding Bahdanau Attention to an Encoder-Decoder LSTM in Keras requires four structural steps:

1. **Keep All Encoder States:**
   Set `return_sequences=True` (alongside `return_state=True`) on the encoder `LSTM` so that every input word's hidden state vector is preserved in `encoder_outputs`.
2. **Compute Attention:**
   Pass `[decoder_outputs, encoder_outputs]` as `[query, value]` into `tf.keras.layers.AdditiveAttention()`. *(When `key` is omitted, Keras automatically uses `value` as both key and value).*
3. **Concatenate Context:**
   Merge the resulting `context_vector` with `decoder_outputs` along the feature axis via `layers.Concatenate(axis=-1)` (doubling the feature dimension from `LATENT_DIM` to `LATENT_DIM * 2`).
4. **Predict Output:**
   Feed the combined tensor into the final `Dense(vocab_size, activation="softmax")` layer.

---

## 5. Complete Hands-On Code (English-to-Hindi NMT)

### Step 1: Data Cleaning & Vectorization

```python
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model

# Load dataset
data = pd.read_csv("/content/Dataset_English_Hindi.csv")

def clean_english(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    return text.strip()

def clean_hindi(text):
    text = str(text).strip()
    # Retain Devanagari Unicode range (\u0900-\u097F) and punctuation
    text = re.sub(r"[^\u0900-\u097F?.!,]+", " ", text)
    return f"[start] {text.strip()} [end]"

eng_texts = data['English'].apply(clean_english).tolist()
hin_texts = data['Hindi'].apply(clean_hindi).tolist()

# Vectorizers
MAX_VOCAB = 15000
MAX_LEN = 30

eng_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB, output_sequence_length=MAX_LEN
)
hin_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=MAX_LEN + 1,
    standardize=None
)

eng_vectorizer.adapt(eng_texts)
hin_vectorizer.adapt(hin_texts)

# Teacher Forcing shifted inputs and targets
encoder_input_data = eng_vectorizer(eng_texts)
hin_sequences = hin_vectorizer(hin_texts)

decoder_input_data = hin_sequences[:, :-1]   # Everything except [end]
decoder_target_data = hin_sequences[:, 1:]   # Everything except [start]
```

### Step 2: Training Model with `AdditiveAttention`

```python
EMB_DIM = 256
LATENT_DIM = 512

eng_vocab_size = len(eng_vectorizer.get_vocabulary())
hin_vocab_size = len(hin_vectorizer.get_vocabulary())

# --- ENCODER ---
encoder_inputs = layers.Input(shape=(None,), name="encoder_inputs")
enc_emb = layers.Embedding(eng_vocab_size, EMB_DIM, mask_zero=False)(encoder_inputs)

# Step 1: return_sequences=True preserves hidden states for EVERY input time step
encoder_lstm = layers.LSTM(
    LATENT_DIM, return_sequences=True, return_state=True, name="encoder_lstm"
)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# --- DECODER ---
decoder_inputs = layers.Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = layers.Embedding(hin_vocab_size, EMB_DIM, mask_zero=False)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = layers.LSTM(
    LATENT_DIM, return_sequences=True, return_state=True, name="decoder_lstm"
)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# --- ATTENTION LAYER ---
# Step 2: AdditiveAttention takes [query, value] -> [decoder_outputs, encoder_outputs]
attention_layer = layers.AdditiveAttention(name="attention_layer")
context_vector = attention_layer([decoder_outputs, encoder_outputs])

# Step 3: Concatenate decoder outputs and context vector along the last axis
# Shape becomes (batch_size, target_seq_len, LATENT_DIM * 2)
decoder_combined_context = layers.Concatenate(
    axis=-1, name="concat_layer"
)([decoder_outputs, context_vector])

# Step 4: Softmax Output Layer
decoder_dense = layers.Dense(
    hin_vocab_size, activation="softmax", name="decoder_dense"
)
decoder_outputs = decoder_dense(decoder_combined_context)

# --- FULL TRAINING MODEL ---
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train model
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)
```

### Step 3: Standalone Inference Pipeline

During inference, the standalone encoder must return **all** `encoder_outputs` alongside `[state_h, state_c]` so the decoder can query them at every generation step:

```python
# 1. Standalone Encoder Model (outputs full sequence + final states)
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# 2. Standalone Decoder Model
decoder_state_input_h = layers.Input(shape=(LATENT_DIM,))
decoder_state_input_c = layers.Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Input for full sequence of encoder hidden states
encoder_outputs_input = layers.Input(shape=(None, LATENT_DIM))

dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]

# Re-apply trained attention and concatenation layers
context_vector_inf = attention_layer([decoder_outputs_inf, encoder_outputs_input])
decoder_combined_inf = layers.Concatenate(axis=-1)(
    [decoder_outputs_inf, context_vector_inf]
)
decoder_outputs_inf = decoder_dense(decoder_combined_inf)

decoder_model = Model(
    [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf,
)

# Vocabulary lookup maps
hin_vocab = hin_vectorizer.get_vocabulary()
hin_word_to_idx = {word: idx for idx, word in enumerate(hin_vocab)}
hin_idx_to_word = {idx: word for idx, word in enumerate(hin_vocab)}

# 3. Autoregressive Translation Function
def translate_sentence(input_sentence):
    cleaned = clean_english(input_sentence)
    input_seq = eng_vectorizer([cleaned])

    # Unpack encoder_outputs along with h and c states
    enc_outs, h, c = encoder_model.predict(input_seq, verbose=0)
    states_value = [h, c]

    target_seq = np.array([[hin_word_to_idx["[start]"]]])
    decoded_words = []

    for _ in range(MAX_LEN):
        # Pass enc_outs into decoder_model at every step
        output_tokens, h, c = decoder_model.predict(
            [target_seq, enc_outs] + states_value, verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin_idx_to_word.get(sampled_token_index, "")

        if sampled_word == "[end]" or sampled_word == "":
            break

        decoded_words.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return " ".join(decoded_words)

# Example Usage:
print(translate_sentence("How are you?"))  # Output: आप कैसे हो?
```

---

## 6. Summary & Key Takeaways

| Aspect | Traditional Seq2Seq | Seq2Seq + Bahdanau Attention |
| :--- | :--- | :--- |
| **Encoder Output** | Only final hidden state ($h_{T_x}, c_{T_x}$) | All intermediate hidden states ($h_1, \dots, h_{T_x}$) |
| **Context Vector** | Single static vector shared across all steps | Dynamic, tailored context vector $c_i$ per step |
| **Long Sequences** | Suffers from the Information Bottleneck | Retains early word context via direct shortcuts |
| **Word Reordering** | Struggles when source & target syntax differ | Softmax weights dynamically align words across positions |
| **Keras Key Change** | `LSTM(..., return_state=True)` | `LSTM(..., return_sequences=True, return_state=True)` + `AdditiveAttention()` |