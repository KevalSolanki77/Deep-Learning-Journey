# Day 5 : Vanishing & Exploding Gradient Problems

## Vanishing Gradient Problem

### Link : [NoteBook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-05-Vanishing-Gradient-Problem/main.ipynb)

### What It Is And Why It Happens

During backpropagation, the gradient for an early layer's weights is a **product** of the gradients from every later layer (chain rule, see Day 4). Multiply 10+ small factors together and the result shrinks toward zero fast. The gradient reaching the early layers becomes tiny, their weights barely update, and in the worst case those layers stop learning entirely.

Sigmoid is the classic culprit. Its derivative is:

$$\sigma'(z) = \sigma(z)\,(1-\sigma(z))$$

This peaks at **0.25** (at `z = 0`) and sits lower everywhere else. Stack 10 sigmoid layers and the chain-rule product includes at least ten factors under 0.25 each: even in the best case, `0.25^10 ≈ 9.5 × 10⁻⁷`. Sigmoid isn't a bad function in general. Its derivative is bounded in a range that collapses under repeated multiplication, and depth is what forces that multiplication to happen.

### The Experiment: Building a Network Deep Enough to Show It

Dataset: `make_moons`, a 2D binary classification problem that isn't linearly separable. It needs a real network to solve, which makes "not learning" easy to distinguish from "doesn't need to learn."

```python
X, y = make_moons(n_samples = 1000, noise = 0.1, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

An 11-hidden-layer, all-sigmoid network, 10 units per layer, sigmoid output:

```python
model = Sequential()

model.add(Dense(10, activation = 'sigmoid', input_dim = 2))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics=['accuracy'])
```

### How To Detect It

```python
old_weights = model.get_weights()[0]

model.fit(X_train, y_train, epochs = 10)

new_weights = model.get_weights()[0]

gradient = (old_weights - new_weights) / 0.001
percent_change = abs(100 * (old_weights - new_weights) / old_weights)

print(f"Gradient :\n {gradient}")
print(f"\n Change In Weights(%) :\n {percent_change}")
```

Dividing by `0.001` reverse-engineers an approximate effective gradient from total weight movement, using Adam's default learning rate. Adam adjusts its own step size internally, so this isn't the literal per-step gradient, but it works as a diagnostic: a weight that barely moved after 10 full epochs was driven by a tiny gradient.

**Result, the 12-layer sigmoid network:**

```text
Change In Weights(%) :
 [[ 2.16  0.50  1.20  0.31  3.20 55.69  0.59  0.60  0.62  0.17]
  [ 7.74  0.13  1.56  0.45  0.90  1.35  0.38  0.79  0.11  0.14]]
```

Nearly every first-layer weight moved under 3% across 10 full epochs (one outlier neuron aside). The training log backs this up: accuracy sat at **0.51** for all 10 epochs and loss stayed pinned near **0.693**, which is `ln(2)`, the loss value for a binary classifier that guesses at random. A loss frozen at `ln(2)` on a binary task is a specific, reliable tell that gradients aren't reaching the network.

## 5 Ways To Handle Vanishing Gradients

### 1. Reduce Model Complexity (Implemented)

Fewer layers means fewer multiplicative terms in the chain-rule product, so the gradient shrinks less on its way back. Same dataset, same sigmoid activation, 3 hidden layers instead of 11:

```python
model = Sequential()

model.add(Input(shape = (2,)))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

old_weights = model.get_weights()[0]
model.fit(X_train, y_train, epochs = 10)
new_weights = model.get_weights()[0]

gradient = (old_weights - new_weights) / 0.001
percent_change = abs(100 * (old_weights - new_weights) / old_weights)
```

**Result:** accuracy climbed from 0.49 to 0.76 over the 10 epochs and loss dropped from 0.716 to 0.688. Modest, but real movement, unlike the flat 0.51/0.693 above. Weight changes now range from **13% to 380%** instead of sitting under 3%. Cutting the layer count was enough on its own to let gradients reach the first layer.

### 2. Use Different Activation Functions, ReLU (Implemented)

This experiment holds depth constant at the original 11 hidden layers, the exact depth that failed with sigmoid, and swaps sigmoid for ReLU in the hidden layers only (the output stays sigmoid, since this is binary classification):

```python
model = Sequential()

model.add(Input(shape = (2,)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

old_weights = model.get_weights()[0]
model.fit(X_train, y_train, epochs = 10)
new_weights = model.get_weights()[0]

gradient = (old_weights - new_weights) / 0.001
percent_change = abs(100 * (old_weights - new_weights) / old_weights)
```

**Result:** accuracy went from 0.70 to **0.975**, loss dropped from 0.688 to **0.087**. This network learns. Weight changes land in a healthy 9% to 67% range. Depth was held fixed at the exact size that failed under sigmoid, so this isolates the activation function as the fix rather than depth. ReLU's derivative is either 0 or exactly 1 for active neurons, so it never gets squashed into a shrinking range the way sigmoid's does.

### 3. Proper Weight Initialization (Not Yet Implemented)

"Xavier initialization" and "Glorot initialization" name the same method, after the same person, Xavier Glorot. One technique, not two.

What it does: instead of starting every weight at the same constant, it draws initial weights from a distribution whose variance is scaled by the layer's fan-in and fan-out (roughly `1/fan_in`), so activations and gradients start out neither too small nor too large before training begins. Glorot/Xavier initialization targets sigmoid/tanh networks. **He initialization**, a related but distinct technique named after Kaiming He, is the ReLU counterpart, with a different variance scale (`2/fan_in`). Worth learning both together since they solve the same problem for different activation functions.

### 4. Batch Normalization (Not Yet Implemented)

Your framing, a layer type, is right. Fuller picture: it's a layer inserted between other layers that normalizes its input batch to zero mean and unit variance, then applies its own learnable scale and shift so the network can undo that normalization where a layer needs to. Keeping activations in a consistent, well-scaled range throughout training stops them, and their gradients, from drifting into the saturated, near-zero-derivative region of activations like sigmoid. That's the widely cited practical reason it helps with vanishing gradients, even though the original paper's stated motivation was reducing "internal covariate shift," the layer's input distribution shifting as earlier layers update during training. Fine to leave as a "learn properly later" item; it's a full topic on its own.

### 5. Residual Networks / Skip Connections (Not Yet Implemented)

It originates from ResNet, a CNN architecture, but the residual/skip-connection idea is a general building block used anywhere networks get very deep, including Transformers, which We'll hit later in the LLM stage.

The mechanism: instead of a block computing `output = F(x)`, a residual block computes `output = F(x) + x`, an identity shortcut that adds the block's input straight to its output. The derivative of that added `x` term with respect to itself is 1, so gradients get an additive path back to earlier layers that bypasses the multiplicative chain-rule product entirely. That's the fix: it doesn't make each multiplicative factor bigger, it gives the gradient a second route that isn't multiplicative at all.

## Exploding Gradient Problem (Brief)

The opposite failure mode, same root cause of repeated multiplication through many layers, diverging instead of shrinking. It shows up mostly in **RNNs**, where the same weight matrix applies repeatedly across time steps. If that matrix's values sit above 1, the repeated multiplication compounds instead of decaying, gradients grow large, and weight updates overshoot, often surfacing as `NaN` losses.

**Fix: gradient clipping.** Cap the gradient before it updates weights, so one runaway backward pass can't blow up the whole network. Basic mechanism for now, to fill in later: clipping usually rescales the entire gradient vector when its norm exceeds a threshold, preserving direction while shrinking magnitude, rather than clamping each value on its own.


### Link : [NoteBook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-05-Vanishing-Gradient-Problem/main.ipynb)