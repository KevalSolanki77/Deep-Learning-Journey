# Day 12 - Weight Initialization

## Reference Table

| Initializer | String Alias | Activation Function Match | Mathematical Distribution |
| :--- | :--- | :--- | :--- |
| **Glorot Uniform (Xavier)** | `'glorot_uniform'` | `sigmoid`, `tanh` | Uniform within \(\pm\sqrt{6 / (\text{fan\_in} + \text{fan\_out})}\) |
| **Glorot Normal (Xavier)** | `'glorot_normal'` | `sigmoid`, `tanh` | Normal centered at 0 with \(\sigma = \sqrt{2 / (\text{fan\_in} + \text{fan\_out})}\) |
| **He Normal** | `'he_normal'` | `relu`, `leaky_relu` | Truncated normal with \(\sigma = \sqrt{2 / \text{fan\_in}}\) |
| **He Uniform** | `'he_uniform'` | `relu`, `leaky_relu` | Uniform within \(\pm\sqrt{6 / \text{fan\_in}}\) |
| **Random Normal** | `'random_normal'` | Generic baseline | Normal distribution (requires manual mean/stddev tuning) |
| **Constant / Zeros** | `'zeros'`, `'ones'` | *Avoid for kernels* (Fine for biases) | Fills the tensor entirely with a single scalar value |

# What Not to Do

## Case 1: Zero Initialization

Setting every weight and bias to zero before training.

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='relu'),
    Dense(1, activation='sigmoid')
])

initial_weights = model.get_weights()
initial_weights[0] = np.zeros(shape=initial_weights[0].shape)
initial_weights[1] = np.zeros(shape=initial_weights[1].shape)
initial_weights[2] = np.zeros(shape=initial_weights[2].shape)
initial_weights[3] = np.zeros(shape=initial_weights[3].shape)

model.set_weights(initial_weights)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=100, validation_split=0.2)
```

**Problem.** Training loss sits at 0.6932 (`ln(2)`, the loss of a model that guesses at random) for all 100 epochs, and accuracy stays pinned near 0.50. After training, `model.get_weights()` still shows every weight at exactly 0.0, only the output bias moved, from 0.0 to 0.00358838.

The root cause is that every neuron in a layer starts with the identical weight vector (all zeros), so every neuron in that layer computes the identical output, receives the identical gradient during backpropagation, and gets updated by the identical amount. The whole layer behaves as one neuron no matter how many units it has, and since the gradient computed from an all-zero weight matrix is itself zero in this setup, the weights never move at all. No training happens.

## Case 1b: Zero Initialization with Sigmoid

Same zero-weight setup, sigmoid activations in both layers instead of ReLU.

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
# same zero-initialization and training as above
```

**Problem.** This time the weights do move, `model.get_weights()[1]` ends at `[-0.23794529, -0.23794529]`, but both values in that pair are identical to six decimal places. Every weight in the first hidden layer's kernel also moves in lockstep across its two neurons: `[0.03817702, 0.03817702]` and `[0.12182502, 0.12182502]`. The two neurons started identical and trained identical, so despite having two units, that layer still computes one function. Mathematically the network collapses to a single-neuron hidden layer, which means it can only draw a straight-line decision boundary, visible directly in the decision regions plot: the entire input space gets assigned to one class.

This failure mode has a name: **symmetry**. Any initialization scheme that gives two neurons in the same layer identical starting weights makes them redundant for the rest of training, no matter what activation function sits on top. Zero initialization is the most extreme case of this, but it isn't the only way to trigger it.

## Case 2: Non-Zero Constant Value

Every weight set to 0.5 instead of 0.

```python
initial_weights[0] = np.ones(shape=initial_weights[0].shape) * 0.5
initial_weights[1] = np.ones(shape=initial_weights[1].shape) * 0.5
initial_weights[2] = np.ones(shape=initial_weights[2].shape) * 0.5
initial_weights[3] = np.ones(shape=initial_weights[3].shape) * 0.5
```

```text
Before Training Weights:
[array([[0.5, 0.5], [0.5, 0.5]]), array([0.5, 0.5]), array([[0.5], [0.5]]), array([0.5])]

After Training Weights:
[array([[0.5336862, 0.5336862], [0.57626563, 0.57626563]]), array([0.19560882, 0.19560882]),
 array([[0.02261607], [0.02261607]]), array([-0.01622712])]
```

**Problem.** Swapping zero for 0.5 changes nothing about the underlying issue. Both neurons in the hidden layer still start identical, so they still receive identical gradients and still train identical, `0.5336862` appears twice in the first row, `0.57626563` twice in the second. The value you pick as the constant is irrelevant. What breaks training is every weight in a layer starting at the *same* value, zero or not.

## Case 3: Random Initialization, Small Values

Four hidden tanh layers, weights drawn from `N(0, 1) * 0.01`, a standard small-random-value scheme.

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='tanh'),
    Dense(2, activation='tanh'),
    Dense(2, activation='tanh'),
    Dense(2, activation='tanh'),
    Dense(1, activation='sigmoid')
])

initial_weights[0] = np.random.randn(2, 2) * 0.01
# ... same pattern for every kernel, biases set to zero
```

```text
Before: [[ 0.00801897, -0.01287528], [-0.01070357,  0.00568523]]
After:  [[ 0.007916  , -0.01284557], [-0.01075947,  0.00571006]]
```

**Problem.** Random breaks symmetry, the two neurons in each layer now start different and train different, which already fixes Case 1 and Case 2's failure. But scaling every weight down to roughly ±0.01 trades one problem for another. The first-layer weights after 100 epochs are barely different from where they started (`0.00801897` → `0.007916`), which is the vanishing gradient problem from Day 5 showing up again, caused this time by the initialization scale rather than network depth. Four stacked tanh layers, each with a derivative capped at 1, compound with tiny starting weights to produce tiny pre-activations, and tiny pre-activations sit in the region of tanh/sigmoid where the function is closest to linear and the gradient is smallest to begin with. For ReLU under the same small-value scheme, the symptom looks different but the cause is the same: convergence stretches out to thousands of epochs instead of stalling outright.

## Case 4: Random Initialization, Large Values

No code was run for this case, the notebook states the expected failure mode directly rather than demonstrating it:

- **Tanh / Sigmoid**: large pre-activations push the input deep into the activation function's flat tails, where the output saturates at the function's maximum or minimum and the local gradient is near zero regardless of what the input actually is.
- **ReLU**: has no upper bound, so large weights produce large positive pre-activations, and the gradient scales with them. Instead of vanishing, the gradient grows unstably large, Day 5's exploding gradient problem.

## What All Four Cases Have in Common

Two separate failure modes, not one:

- **Identical weights within a layer** (Cases 1 and 2) break the network through symmetry, regardless of what the shared value is.
- **Wrong weight scale** (Cases 3 and 4), too small or too large, breaks the network through vanishing or exploding gradients, even once symmetry is already broken by randomness.

A correct scheme needs both: different starting values per neuron, and a principled scale tied to the layer's size. That's exactly what Xavier/Glorot and He initialization provide.

# Initialization Techniques That Work

Same experimental setup throughout: four hidden layers of 2 units, `make_circles`, 100 epochs, `validation_split=0.2`, only the activation and `kernel_initializer` change.

## Xavier / Glorot Initialization (for Sigmoid / Tanh)

Designed around a specific goal: keep the variance of activations roughly constant as they pass forward through the network, and keep the variance of gradients roughly constant as they pass backward. Achieving both at once means the scale has to account for a layer's `fan_in` (how many inputs feed into each neuron) and `fan_out` (how many outputs each neuron feeds), not just one or the other.

**Glorot Normal.** Draws weights from a normal distribution centered at 0:

$$
w \sim \mathcal{N}\left(0,\sigma^2 I\right), \qquad \sigma = \sqrt{\frac{2}{\mathrm{fan\_in}}}
$$

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(2, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(2, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(2, activation='tanh', kernel_initializer='glorot_normal'),
    Dense(1, activation='sigmoid')
])
```


**Glorot Uniform.** Same variance target, drawn from a uniform distribution instead of a normal one:

$$
W \sim U\left(-\sqrt{\frac{6}{\mathrm{fan_in}+\mathrm{fan_out}}},\sqrt{\frac{6}{\mathrm{fan_in}+\mathrm{fan_out}}}\right)
$$

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(2, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(2, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(2, activation='tanh', kernel_initializer='glorot_uniform'),
    Dense(1, activation='sigmoid')
])
```

Both produce an actual decision boundary instead of the single flat prediction zero-initialization gave Case 1. Worth being precise about what "better" means here: the boundary each one draws is still close to a straight or gently curved line, not the ring shape the data actually needs. With only 2 units per hidden layer and 100 epochs, this network is too small and too briefly trained to fit `make_circles` well regardless of initialization scheme; what the comparison against Case 1 shows is that the network is learning something at all, not that it has solved the task.

## He Initialization (for ReLU / Leaky ReLU)

Glorot's derivation assumes an activation function that's roughly linear and symmetric around zero, true for tanh near the origin, not true for ReLU, which zeroes out half its input range entirely. He initialization accounts for that: since ReLU kills roughly half the pre-activations outright, it doubles the variance target so the surviving half still carries enough signal forward.

**He Normal.**

$$
W \sim \mathcal{N}\left(0,\sigma^2\right), \qquad \sigma = \sqrt{\frac{2}{\mathrm{fan_in}}}
$$

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='relu', kernel_initializer='he_normal'),
    Dense(2, activation='relu', kernel_initializer='he_normal'),
    Dense(2, activation='relu', kernel_initializer='he_normal'),
    Dense(2, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid')
])
```

**He Uniform.**

$$
W \sim U\left(-\sqrt{\frac{6}{\mathrm{fan_in}}},\sqrt{\frac{6}{\mathrm{fan_in}}}\right)
$$

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='relu', kernel_initializer='he_uniform'),
    Dense(2, activation='relu', kernel_initializer='he_uniform'),
    Dense(2, activation='relu', kernel_initializer='he_uniform'),
    Dense(2, activation='relu', kernel_initializer='he_uniform'),
    Dense(1, activation='sigmoid')
])
```

He Uniform produces a visible diagonal decision boundary, similar in character to the Glorot results. He Normal in this particular run collapsed to predicting one class across the entire input space, the same symptom Case 1's zero-init showed, even though the weights here started different per neuron and symmetry wasn't the issue this time. With only 2 ReLU units per layer, a handful of dead neurons (Day 11's dying ReLU problem) is enough to flatten most of a 4-layer, 2-unit-wide path, and this result is one specific random seed's outcome, not evidence that He Normal is unreliable in general. Worth rerunning with a different `random_state` or a wider layer to see whether this was that specific run or a repeatable pattern.
