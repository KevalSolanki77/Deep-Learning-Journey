# Day 29: Sequence-to-Sequence (Encoder-Decoder) Architecture

A comprehensive guide covering the **Encoder-Decoder (Seq2Seq)** paradigm, why conventional recurrent models fail when input and output lengths diverge, detailed architectural mechanics (Teacher Forcing vs. Inference), the Information Bottleneck problem, and a look ahead to the **Attention Mechanism**.

## Table of Contents

1. [What is an Encoder-Decoder?](#1-what-is-an-encoder-decoder)

2. [Why We Need It & What Problem It Solves](#2-why-we-need-it--what-problem-it-solves)

   * [The Rigid Length Constraint of Vanilla RNNs](#the-rigid-length-constraint-of-vanilla-rnns)

   * [The Variable-Length Translation Challenge](#the-variable-length-translation-challenge)

3. [Detailed Architecture & Workflow](#3-detailed-architecture--workflow)

   * [The Encoder](#a-the-encoder)

   * [The Context Vector (The Latent Bridge)](#b-the-context-vector-the-latent-bridge)

   * [The Decoder](#c-the-decoder)

   * [Training Paradigm: Teacher Forcing](#d-training-paradigm-teacher-forcing)

   * [Inference Paradigm: Autoregressive Decoding](#e-inference-paradigm-autoregressive-decoding)

4. [Code Implementation (English-to-Hindi Neural Machine Translation)](#4-code-implementation-english-to-hindi-neural-machine-translation)

   * [Data Preprocessing & Special Tokens](#step-1-preprocessing--special-tokens)

   * [Model Assembly (Keras Functional API)](#step-2-model-assembly)

   * [Inference Pipeline](#step-3-inference-pipeline)

5. [The Critical Flaw: The Information Bottleneck](#5-the-critical-flaw-the-information-bottleneck)

6. [Summary & Key Takeaways](#6-summary--key-takeaways)

## 1. What is an Encoder-Decoder?

The **Encoder-Decoder** (often called **Sequence-to-Sequence** or **Seq2Seq**) is a neural network architecture designed to convert an input sequence from one domain into an output sequence in another domain, where the two sequences can have **arbitrary and unequal lengths** ($T_x \neq T_y$).

First popularized by Sutskever et al. (2014) and Cho et al. (2014), the framework decomposes sequence transduction into two distinct stages:

1. **The Encoder:** Reads the entire variable-length source sequence token-by-token and condenses its semantic essence into a fixed-size mathematical representation called the **Context Vector**.

2. **The Decoder:** Takes the context vector as its initial state and generates the target sequence step-by-step, conditioning each new word on both the context and previously generated tokens.

## 2. Why We Need It & What Problem It Solves

### The Rigid Length Constraint of Vanilla RNNs

Standard Recurrent Neural Networks (and standard LSTMs/GRUs) inherently assume a 1-to-1 temporal mapping between input steps and output steps:

* **Synced Many-to-Many:** Standard RNNs emit an output $\hat{y}_t$ for every single input $x_t$. The output sequence length $T_y$ is strictly bound to the input sequence length $T_x$ ($T_x = T_y$).

* **Classification (Many-to-One):** Ingests $T_x$ steps and produces a single prediction.

* **Generation (One-to-Many):** Takes a single static prompt and expands it into $T_y$ steps.

### The Variable-Length Translation Challenge

In real-world sequence transduction tasks, input and output lengths almost never match:

* **Machine Translation:**

  * English: *"How are you?"* ($3$ words)

  * Hindi: *"आप कैसे हैं?"* ($3$ words)

  * English: *"I am learning deep learning."* ($5$ words)

  * Hindi: *"मैं डीप लर्निंग सीख रहा हूँ।"* ($6$ words)

* **Text Summarization:** Ingests a 500-word article ($T_x = 500$) and produces a 50-word summary ($T_y = 50$).

* **Speech-to-Text (ASR):** Ingests thousands of acoustic wave frames ($T_x = 2500$) and produces a few words ($T_y = 7$).

* **Question Answering:** A long paragraph plus question yields a concise answer.

**What Problem Seq2Seq Solves:** It decouples input sequence reading from output sequence generation. The model does not need to guess the output while still reading the input; it ingests the entire source context first before producing a single token of the target.

## 3. Detailed Architecture & Workflow

![Encoder Decoder Architecture](https://i.ibb.co/gLDtrBmt/Working-of-Encoder-Decoder.webp)

### A. The Encoder

* Receives the sequence of source tokens: $X = (x_1, x_2, \dots, x_{T_x})$.

* Passes tokens through an **Embedding Layer** to produce dense vectors.

* The recurrent unit (typically LSTM) updates its hidden states step-by-step:

$$
h_t^{(enc)}, c_t^{(enc)} = \text{LSTM}_{enc}(e(x_t), h_{t-1}^{(enc)}, c_{t-1}^{(enc)})
$$

* The intermediate step outputs are discarded (`return_sequences=False` or ignored). Only the final states at the terminal step $T_x$ are retained.

### B. The Context Vector (The Latent Bridge)

* The final internal memory states of the encoder form the **Context Vector**:

$$
\mathbf{C} = \left[ h_{T_x}^{(enc)}, c_{T_x}^{(enc)} \right]
$$

* This vector acts as a compressed semantic summary of the entire source sentence.

* It is passed directly to the Decoder as its initial hidden state:

$$
h_0^{(dec)} = h_{T_x}^{(enc)}, \quad c_0^{(dec)} = c_{T_x}^{(enc)}
$$

### C. The Decoder

* A recurrent network tasked with predicting the next target word given the context and all preceding words:

$$
P(y_t \mid y_1, \dots, y_{t-1}, X) = \text{Softmax}\left(W_v h_t^{(dec)} + b_v\right)
$$

* Two special control tokens manage sequence boundaries:

  * `[start]` / `<SOS>`: Signals the decoder to begin generation.

  * `[end]` / `<EOS>`: Tells the decoder that generation is complete.

### D. Training Paradigm: Teacher Forcing

During training, if the decoder makes an error at step $t=1$, feeding that mistake into step $t=2$ causes compounding errors that destabilize convergence.

To prevent this, Seq2Seq uses **Teacher Forcing**:

* The decoder does not receive its own previous prediction as input.

* Instead, it receives the **ground-truth target sequence** shifted by one time step:

  * **Decoder Input:** `[start], y_1, y_2, ..., y_{T_y-1}` (Everything except `[end]`)

  * **Decoder Target:** `y_1, y_2, ..., y_{T_y}, [end]` (Everything except `[start]`)

### E. Inference Paradigm: Autoregressive Decoding

At test time, ground-truth target tokens are unavailable. The model must run **autoregressively**:

1. Pass the input sentence through the Encoder to extract the context states $[h, c]$.

2. Pass `[start]` and $[h, c]$ into the Decoder to predict the first word $\hat{y}_1$.

3. Feed $\hat{y}_1$ along with the updated states $[h_1, c_1]$ back into the Decoder to predict $\hat{y}_2$.

4. Repeat step-by-step until the model predicts the `[end]` token or reaches a predefined maximum length.

## 4. Code Implementation (English-to-Hindi Neural Machine Translation)

Based on the standalone Keras functional architecture:

### Step 1: Preprocessing & Special Tokens

```
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Clean strings and wrap target sentences with [start] and [end]
def clean_english(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    return text.strip()

def clean_hindi(text):
    text = str(text).strip()
    # Retain Devanagari Unicode range (\u0900-\u097F)
    text = re.sub(r"[^\u0900-\u097F?.!,]+", " ", text)
    return f"[start] {text.strip()} [end]"

eng_texts = [clean_english(t) for t in raw_eng_list]
hin_texts = [clean_hindi(t) for t in raw_hin_list]

# Vectorization Layers
MAX_VOCAB = 15000
MAX_LEN = 30

eng_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=MAX_LEN
)
hin_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=MAX_LEN + 1,
    standardize=None
)

eng_vectorizer.adapt(eng_texts)
hin_vectorizer.adapt(hin_texts)

# Split Targets for Teacher Forcing
encoder_input_data = eng_vectorizer(eng_texts)
hin_sequences = hin_vectorizer(hin_texts)

decoder_input_data = hin_sequences[:, :-1]  # Exclude [end]
decoder_target_data = hin_sequences[:, 1:]  # Exclude [start]

```

### Step 2: Model Assembly

```
EMB_DIM = 256
LATENT_DIM = 512

eng_vocab_size = len(eng_vectorizer.get_vocabulary())
hin_vocab_size = len(hin_vectorizer.get_vocabulary())

# --- ENCODER ---
encoder_inputs = layers.Input(shape=(None,), name="encoder_inputs")
enc_emb = layers.Embedding(eng_vocab_size, EMB_DIM)(encoder_inputs)

encoder_lstm = layers.LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]  # Context Vector

# --- DECODER ---
decoder_inputs = layers.Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = layers.Embedding(hin_vocab_size, EMB_DIM)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = layers.LSTM(
    LATENT_DIM, return_sequences=True, return_state=True, name="decoder_lstm"
)
# Initialize decoder with Encoder's context vector
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = layers.Dense(hin_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# --- TRAINING GRAPH ---
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
training_model.summary()

training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)

```

### Step 3: Inference Pipeline

During inference, we unpack the trained layers into two distinct execution graphs:

```
# 1. Standalone Encoder: Maps source text to context vector
encoder_model = Model(encoder_inputs, encoder_states)

# 2. Standalone Decoder: Generates one word at a time
decoder_state_input_h = layers.Input(shape=(LATENT_DIM,))
decoder_state_input_c = layers.Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

# 3. Autoregressive Loop
hin_vocab = hin_vectorizer.get_vocabulary()
hin_idx_to_word = {idx: word for idx, word in enumerate(hin_vocab)}
hin_word_to_idx = {word: idx for idx, word in enumerate(hin_vocab)}

def translate_sentence(input_sentence):
    cleaned = clean_english(input_sentence)
    input_seq = eng_vectorizer([cleaned])
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.array([[hin_word_to_idx["[start]"]]])
    decoded_words = []

    for _ in range(MAX_LEN):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )
        sampled_id = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin_idx_to_word.get(sampled_id, "")

        if sampled_word == "[end]" or sampled_word == "":
            break

        decoded_words.append(sampled_word)
        target_seq = np.array([[sampled_id]])
        states_value = [h, c]

    return " ".join(decoded_words)

print(translate_sentence("How are you?"))

```

## 5. The Critical Flaw: The Information Bottleneck

While the basic Encoder-Decoder architecture enables variable-length sequence transduction, it has a major structural limitation known as the **Information Bottleneck**:

$$
X = (x_1, x_2, x_3, \dots, x_{100}) \xrightarrow{\text{squashed into}} \mathbf{C} \in \mathbb{R}^{d}
$$

1. **Fixed-Dimensional Compression:** Whether a sentence contains $3$ words or $80$ words, the encoder is forced to compress all semantic content, grammatical dependencies, and factual nuance into a **single, fixed-size vector** (e.g., $d = 512$).

2. **Catastrophic Performance Degradation on Long Sequences:**

   * For short sentences ($< 15$ words), the context vector retains enough information for accurate translation.

   * As sentences exceed $20\text{--}30$ words, BLEU scores decline significantly.

3. **Recency Bias:** The context vector is simply the final state ($h_{T_x}$). The hidden state tends to be dominated by the last few words read, while earlier tokens from the beginning of the sentence fade due to gradient and memory decay.

## 6. Summary & Key Takeaways

1. **Seq2Seq Decouples Dimensions:** It breaks the $T_x = T_y$ limitation of vanilla RNNs, enabling variable-input to variable-output mapping.

2. **Two Roles, One Model:** The Encoder constructs a representation; the Decoder acts as a conditioned language generator.

3. **Teacher Forcing Accelerates Training:** Feeding ground-truth shifted tokens into the decoder stabilizes training and speeds up convergence.

4. **Inference is Autoregressive:** Generating unseen text requires an iterative loop that feeds the predicted token back in at each step until `[end]` is emitted.

5. **The Bottleneck Mandates Attention:** Condensing long sequences into a single static context vector degrades performance on complex tasks, motivating the step-by-step dynamic weighting of the **Attention Mechanism**.