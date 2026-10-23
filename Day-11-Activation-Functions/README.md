# Day 11 - Activation Functions

## Setup

All experiments below run on Fashion MNIST: 60,000 training images and 10,000 test images, 28×28 grayscale, pixel values normalized to [0, 1].

```python
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255, X_test / 255
```

The test network is the same shape throughout: a `Flatten` input, 10 hidden `Dense` layers of 64 units each, and a 10-unit softmax output. A network this deep is deliberate. It's deep enough that an activation function's effect on gradient flow across layers becomes visible, instead of getting lost in a shallow network where every activation would train fine.

```python
def build_model(activation, depth=10, units=64):
    model = Sequential([Input((28, 28)), Flatten()])
    for _ in range(depth):
        model.add(Dense(units=units, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
```

## 1. Sigmoid

**Details.** Squashes any real input into the range (0, 1). Historically the default choice for hidden layers before ReLU took over, and still standard for a binary classification output layer, where its output doubles as a probability.

**Equation.**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Smooth, differentiable everywhere | Derivative tops out at 0.25, causes vanishing gradients in deep networks |
| Output reads directly as a probability | Not zero-centered: output is always positive, which biases gradient updates in a consistent direction |
| Bounded output stabilizes activations | Expensive per-neuron (`e^{-z}`) compared to ReLU's single comparison |

**Problem it causes: Vanishing Gradient.** Stack enough sigmoid layers and the chain-rule product of many sub-0.25 derivatives shrinks toward zero, so early layers stop receiving a useful gradient. Full experiment with your own measurement below. Small-scale fix: swap sigmoid for ReLU in hidden layers, keep it only where a bounded output is actually needed (gates, binary output).

## Vanishing Gradient Problem: Measuring It

Your code measures this directly instead of just describing it: a callback computes the gradient norm at every layer, before training and after each epoch.

```python
class GradNormLogger(tensorflow.keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y
        self.history = []

    def _norms(self):
        with tensorflow.GradientTape() as tape:
            preds = self.model(self.x, training=False)
            loss = tensorflow.reduce_mean(
                tensorflow.keras.losses.sparse_categorical_crossentropy(self.y, preds))
        grads = tape.gradient(loss, self.model.trainable_variables)
        return [float(tensorflow.sqrt(tensorflow.reduce_mean(tensorflow.square(g)))) for g in grads[::2]]

    def on_train_begin(self, logs=None):
        self.history.append(self._norms())

    def on_epoch_end(self, epoch, logs=None):
        self.history.append(self._norms())
```

`grads[::2]` keeps only the weight gradients and skips the bias gradients, so each entry in the result is one layer's weight-gradient norm.

```python
logger = GradNormLogger(X_train[:256], y_train[:256])
model = build_model("sigmoid", depth=10)
model.fit(X_train, y_train, epochs=10, callbacks=[logger], verbose=0)

norms = np.array(logger.history)
plt.semilogy(norms[-1], marker="o")
plt.xlabel("Layer index (0 = closest to input)")
plt.ylabel("Gradient norm")
```

![Gradient norm by layer, log scale](https://i.ibb.co/3mrHxV0z/vanishing-gradient-plot.png)

Layers 1 through 9 drop steadily on the log scale, the vanishing trend your notes describe: the deeper into the network, the smaller the gradient reaching that layer. Two points worth noting directly rather than smoothing over. Layer 0, the one closest to the input, sits below layer 1 instead of continuing the trend upward, a sign that the very first layer is barely learning at all after 10 sigmoid layers on top of it. Layer 10, the output layer, jumps back up above every hidden layer, which makes sense on its own: it receives the loss gradient directly and never goes through the shrinking chain-rule product the hidden layers do. The 9-layer stretch in between is the part that demonstrates vanishing gradient. The two endpoints are a different story each.

## 2. Tanh

**Details.** A rescaled, zero-centered sigmoid. Was often preferred over sigmoid in hidden layers for exactly that reason, zero-centered outputs keep gradient updates from all pushing in the same direction the way sigmoid's always-positive output does.

**Equation.**

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Zero-centered output, range (-1, 1) | Still saturates: derivative is near zero for large \|z\| |
| Stronger gradients near zero than sigmoid, derivative peaks at 1 instead of 0.25 | Same vanishing-gradient failure mode as sigmoid in deep networks, just slower to arrive |
| Smooth and differentiable everywhere | Still costs an exponential per neuron |

**Problem it causes.** Vanishing gradient again, milder than sigmoid because the derivative's peak is 4× higher, but the same saturation at the tails still strangles gradients once a network gets deep enough. Same fix: ReLU-family activations for hidden layers in deep networks.

## 3. ReLU

**Details.** The modern default for hidden layers. Passes positive inputs through unchanged and zeroes out negative inputs.

**Equation.**

$$f(z) = \max(0, z)$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Derivative is exactly 1 for any positive input, no shrinking across layers | Derivative is exactly 0 for any negative input, a neuron can stop updating entirely |
| Cheap: one comparison, no exponential | Not zero-centered |
| Induces sparsity, negative inputs become exact zeros | Unbounded output can grow large without normalization |

**Problem it causes: Dying ReLU.** If a weight update pushes a neuron's pre-activation negative for every input in the dataset, that neuron outputs zero everywhere, its gradient is zero everywhere, and no further update can ever revive it. It's permanently dead. Full measurement below. Small-scale fix: a smaller learning rate reduces the chance of the one oversized update that kills a neuron in the first place, which is exactly what the experiment below tests.

## Dying ReLU Problem: Measuring It

`dead_fraction` checks one hidden layer's activations across 5,000 samples and flags a neuron as dead if its maximum activation across all of them is exactly zero, meaning it never fired once.

```python
def dead_fraction(model, x, layer_idx=1):
    probe = tensorflow.keras.Model(inputs=model.inputs, outputs=model.layers[layer_idx].output)
    acts = probe(x)
    never_fires = tensorflow.reduce_max(acts, axis=0) == 0
    return tensorflow.reduce_mean(tensorflow.cast(never_fires, tensorflow.float32)).numpy()
```

```python
for lr in [0.001, 0.01, 0.1]:
    m = build_model("relu", depth=10, lr=lr)
    m.fit(X_train, y_train, epochs=10, verbose=0)
    print(lr, dead_fraction(m, X_train[:5000]))
```

| Learning rate | Dead neuron fraction |
|---|---|
| 0.001 | 12.5% |
| 0.01 | 87.5% |
| 0.1 | 84.375% |

The jump from 0.001 to 0.01 backs that up hard, dead neurons go from 1 in 8 to 7 in 8. But 0.01 to 0.1 actually ticks down slightly, 87.5% to 84.375%, not up. The real pattern in this run is a steep rise followed by a plateau, not a clean increase across all three rates. Worth keeping in mind before stating it as a strict monotonic relationship: a higher learning rate raises the risk of dead neurons, but the risk doesn't necessarily keep climbing without bound as the rate climbs further.

## 4. Leaky ReLU

**Details.** ReLU's direct fix for the dying neuron problem: instead of flattening negative inputs to exactly zero, it lets a small, fixed fraction through.

**Equation.**

$$f(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases} \quad \text{(typically } \alpha = 0.01\text{)}$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Gradient is never exactly zero (`α` for negative inputs), so a neuron can't die permanently | `α` is a fixed hyperparameter, not learned, picking it is manual |
| Keeps ReLU's cheap computation | Benefit over plain ReLU isn't consistent across every task |
| Directly solves the dying ReLU failure mode | Still not zero-centered |

**Problem it causes.** Because `α` is fixed rather than learned, it's a guess that may not suit every layer or every dataset equally well. Small-scale fix: let the network learn `α` itself, which is exactly what Parametric ReLU does next.

## 5. Parametric ReLU (PReLU)

**Details.** Leaky ReLU with `α` promoted from a fixed constant to a learnable parameter, trained by backpropagation alongside every other weight, usually one `α` per neuron or per channel.

**Equation.**

$$f(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases} \quad \text{(}\alpha\text{ learned, not fixed)}$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| The negative-side slope adapts to what the data actually needs, rather than a one-size guess | Extra parameters to learn, one or more `α` per layer |
| Removes Leaky ReLU's manual tuning step | More prone to overfitting, especially on small datasets |
| Still solves dying ReLU, same as Leaky ReLU | Slightly more training overhead than a fixed constant |

**Problem it causes.** The extra learned parameters give the network more capacity to memorize rather than generalize, particularly with little training data. Small-scale fix: fall back to a fixed-`α` Leaky ReLU, or add standard regularization (Day 10) if PReLU shows a bigger train/val gap than plain ReLU does on the same data.

## 6. ELU

**Details.** Smooths out the sharp corner Leaky ReLU and PReLU both have at `z = 0`. Positive inputs pass through unchanged; negative inputs saturate smoothly toward `-α` instead of continuing linearly downward.

**Equation.**

$$f(z) = \begin{cases} z & z > 0 \\ \alpha (e^z - 1) & z \leq 0 \end{cases} \quad \text{(typically } \alpha = 1\text{)}$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Smooth and differentiable at `z = 0`, no kink | Costs an exponential for every negative input, pricier than Leaky ReLU's single multiply |
| Negative outputs push the mean activation closer to zero, which speeds up learning the way tanh's zero-centering does | Still a 1-neuron-at-a-time cost even where ReLU would be free |
| Negative-side saturation (rather than a straight line) tends to make the network more robust to noise | Choice of `α` still matters, though less sensitively than Leaky ReLU's |

**Problem it causes.** The exponential on every negative activation adds real compute cost across a large network, the trade for ELU's smoother gradient behavior. Small-scale fix: SELU keeps the same exponential shape but adds a self-normalizing guarantee that can let a plain feedforward network skip batch normalization entirely, worth the same exponential cost for that extra benefit.

## 7. SELU

**Details.** ELU's shape, scaled by two fixed constants (`λ ≈ 1.0507`, `α ≈ 1.6733`) chosen specifically so that, under the right conditions, each layer's output activations converge toward zero mean and unit variance on their own, no separate normalization layer required.

**Equation.**

$$f(z) = \lambda \begin{cases} z & z > 0 \\ \alpha (e^z - 1) & z \leq 0 \end{cases}$$

**Advantages & Disadvantages.**

| Advantages | Disadvantages |
|---|---|
| Self-normalizing: can replace Batch Normalization in a plain feedforward network | The self-normalizing guarantee only holds under specific conditions, break any of them and it's gone |
| Keeps ELU's smooth negative-side saturation | Requires LeCun normal weight initialization specifically |
| Can enable deeper plain networks without extra normalization layers | Requires `AlphaDropout` instead of ordinary `Dropout` if dropout is used at all |

**Problem it causes.** The self-normalizing property is conditional, not automatic. It requires LeCun normal initialization, `AlphaDropout`, and a plain fully-connected architecture. Add Batch Normalization, skip connections, or any non-sequential architecture, and the guarantee no longer holds. Small-scale fix: reserve SELU for genuinely plain feedforward networks built under those exact conditions, and use ELU or ReLU-family activations everywhere else.