# Day 15 - AdaGrad, RMSprop, Adam

## The Shared Idea

Day 14 covered momentum, which changes the *direction* SGD moves by averaging past gradients. This entry covers a separate idea: giving every weight its own step size instead of one global learning rate for the whole network. Each method scales a weight's update by how large that weight's own gradients have historically been, a weight that's been seeing large gradients gets smaller steps, a weight seeing small or infrequent gradients gets larger ones. AdaGrad, RMSprop, and Adam all implement this same core idea; they differ in how they remember gradient history and whether they also carry Day 14's momentum.

| | Memory of squared gradients | Direction memory (momentum) | Bias correction |
|---|---|---|---|
| AdaGrad | Sum of all history | No | Not needed |
| RMSprop | Moving average (forgets) | No | No |
| Adam | Moving average (forgets) | Yes | Yes |

## The Update Rules

All three operate element-wise, independently per weight.

**AdaGrad**

$$G \leftarrow G + g^2 \qquad w \leftarrow w - \frac{\eta}{\sqrt{G} + \epsilon}\, g$$

**RMSprop**

$$s \leftarrow \rho s + (1-\rho) g^2 \qquad w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon}\, g$$

**Adam**

$$m \leftarrow \beta_1 m + (1-\beta_1) g \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$

$$\hat{m} = \frac{m}{1-\beta_1^t} \qquad \hat{v} = \frac{v}{1-\beta_2^t} \qquad w \leftarrow w - \eta\, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

Adam's defaults, used almost everywhere unless explicitly tuned: `lr=1e-3`, `β1=0.9`, `β2=0.999`, `ε=1e-8`.

## How Each One Evolved From the Last

### AdaGrad: the Per-Weight Brake

`G` only accumulates, it never shrinks, so a weight whose gradients have been large or frequent builds a large `G` and gets its steps shrunk accordingly. A weight tied to a rare feature stays with a small `G` and keeps taking full-sized steps. This is the entire appeal for sparse data: a feature that appears once every few thousand examples shouldn't have its one update drowned out by the learning rate tuned for a feature that appears in every example.

Under a constant gradient `g`, `G` after `t` steps equals `t·g²`, so the step size decays like `lr/√t`. The very first step works out to roughly `lr · sign(g)`, independent of how large `|g|` actually is, which is why AdaGrad's first move is insensitive to the raw scale of the loss.

**The flaw.** `G` only ever grows. Over a long training run, especially a deep network, the accumulated sum eventually makes every step size shrink toward zero regardless of whether the weight still needs to move. Training doesn't diverge, it just stalls, the optimizer effectively stops adjusting the weight before the loss has actually converged.

### RMSprop: AdaGrad With Forgetting

Replace the running sum `G` with an exponential moving average `s`, and the monotonic-decay flaw disappears. The moving average has an effective memory window of roughly `1/(1-ρ)` steps, old gradients eventually age out instead of accumulating forever, so the step size settles near `lr` rather than decaying to zero.

**Two quirks worth knowing before using it.** `s` starts at zero, so the very first update is scaled by `1/√(1-ρ)`, which equals about `3.2` for the default `ρ=0.9`, meaning the first step is roughly 3.2x larger than the steps that follow once `s` warms up. Second, near a minimum where the true gradient should be shrinking to zero, `s` still reflects whatever gradient magnitude has been typical recently, so the step size stays near `lr` instead of shrinking with the gradient, which shows up as jitter around the minimum rather than clean settling.

### Adam: RMSprop + Momentum + Bias Correction

Adam keeps RMSprop's `v` for scale, adds Day 14's momentum as `m` for direction, and adds a bias correction that neither earlier method needed. `m` smooths out which direction to move, `√v` normalizes how far to move in that direction, and `m̂`, `v̂` fix a cold-start problem: both `m` and `v` start at zero, so without the `/(1-β^t)` correction, early steps would be biased toward zero exactly when the optimizer has the least information to compensate.

**`m̂/√v̂` behaves like a signal-to-noise ratio.** A gradient that's been pointing the same direction consistently builds a large `m̂` relative to `√v̂`, producing something close to a full-sized step. A gradient that keeps flipping sign, noisy rather than informative, has its `m̂` partially cancel itself out across updates while `√v̂` stays elevated from the squared magnitudes, producing a small step. The ratio is effectively asking "how much of this gradient is signal versus noise" before committing to a step size.

**Each weight moves by roughly at most `lr` per step.** Under a constant gradient, the bias correction removes the cold-start inflation entirely and the step becomes exactly `lr` from the first update onward, verified directly below. This gives `lr` a concrete, literal meaning in Adam: it's approximately the largest distance, in weight units, any single parameter moves in one update.

## Code

The numpy implementations tested against a shared ravine loss, `0.5(x² + 25y²)`, the same surface from Day 14:

```python
def AdaGrad(w0, lr, epochs):
    """Accumulate squared gradients to scale each parameter update."""
    eps = 1e-8
    w = w0.copy()
    G = np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(epochs):
        g = grad(w)
        G += g**2
        w = w - lr * g / (np.sqrt(G) + eps)
        path.append(w.copy())
    return np.array(path)
```

```python
def RMSProp(w0, rho, lr, epochs):
    """Scale updates using an exponential average of squared gradients."""
    eps = 1e-8
    w = w0.copy()
    v = np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(epochs):
        g = grad(w)
        v = rho * v + (1 - rho) * g**2
        w = w - lr * g / (np.sqrt(v) + eps)
        path.append(w.copy())
    return np.array(path)
```

```python
def Adam(w0, beta1, beta2, lr, epochs):
    """Adam with first- and second-moment bias correction."""
    eps = 1e-8
    w = w0.copy()
    m = np.zeros_like(w0)
    v = np.zeros_like(w0)
    path = [w.copy()]
    for t in range(1, epochs + 1):
        g = grad(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)
```

```python
w0 = np.array([10.0, 1.0])
lr, rho, beta1, beta2, steps = 0.05, 0.9, 0.9, 0.999, 500

adagrad = AdaGrad(w0, lr, steps)
rmsp = RMSProp(w0, rho, lr, steps)
adam = Adam(w0, beta1, beta2, lr, steps)
```

**Keras equivalents**, same hyperparameter names mapped directly onto the framework call:

```python
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
```

```python
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,          # Discounting factor for the old gradients
    momentum=0.0,     # Optional momentum multiplier
    epsilon=1e-07
)
```

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,       # Exponential decay rate for the 1st moment estimates
    beta_2=0.999,     # Exponential decay rate for the 2nd moment estimates
    epsilon=1e-07
)
```

Worth noting directly: Keras's `RMSprop` accepts an optional `momentum` argument, left at `0.0` here. Setting it would bolt Day 14's momentum onto RMSprop without Adam's bias correction, a real, usable combination, just a different one than what the math section above derives.

## Worked Numbers

Hand-traceable arithmetic with a constant gradient `g = 3`, `lr = 0.1`, verified by direct simulation rather than taken on faith:

| Optimizer | Step 1 | Step 100 |
|---|---|---|
| AdaGrad | 0.1 | 0.01 |
| RMSprop (ρ=0.9) | 0.316 | ≈ 0.100 |
| Adam (corrected) | 0.1 | 0.1 |
| Adam without correction (β2=0.999) | ~0.32 | still inflated, peaks ~6.57x lr at step 12 |

The uncorrected-Adam row is worth sitting with since it's the clearest demonstration of what bias correction actually buys. Without `/(1-β^t)`, the step ratio to `lr` climbs from 3.16x at step 1 to a peak of 6.57x at step 12, then only partially relaxes, it's still sitting around 3.24x by step 100, nowhere near the `1.0x` the corrected version holds from the very first step. The correction isn't a minor numerical nicety, it's the difference between a step size that means what `lr` says it means and one that's off by 3-6x depending on how far into training you are.

## Real-World Use

| Optimizer | Where it fits |
|---|---|
| AdaGrad | Sparse features: bag-of-words text, one-hot IDs, ad CTR prediction, recommender embeddings. GloVe was trained with it. Not used for modern deep nets. |
| RMSprop | RNNs/LSTMs, reinforcement learning (the original Atari DQN), GANs (the WGAN authors preferred it over momentum-based Adam). |
| Adam / AdamW | The default almost everywhere. AdamW is the standard for transformers and LLMs (β2 ≈ 0.95 for LLM training stability; BERT fine-tuning typically runs lr ≈ 2e-5). DCGAN uses lr=2e-4, β1=0.5. |

**Memory cost.** Adam keeps two extra buffers per weight (`m` and `v`), which adds up fast at scale, about 56 GB of optimizer state for a 7B-parameter model in fp32. SGD with momentum (Day 14) keeps only one buffer per weight. This memory gap is the direct motivation behind 8-bit Adam variants that compress the optimizer state.

**Where SGD+momentum still wins.** Well-tuned, SGD with momentum often generalizes slightly better than Adam on vision tasks like ResNets, the same point Day 14 made from the optimizer-choice side; this entry's methods are the reason that comparison exists at all, since Adam is the thing SGD+momentum is usually being compared against.

## Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss flatlines high (AdaGrad) | Accumulator `G` grew too large | Switch to RMSprop/Adam, or raise `lr` |
| Early loss spike (RMSprop) | Zero-initialized `s` | Smaller `lr` or a warmup schedule |
| Jitter near the minimum (RMSprop) | Step size stays near `lr` instead of shrinking | Decay `lr` |
| NaN or spike (Adam) | `lr` too high | Lower it 3-10x, add warmup |
| Mid-training spike in LLM training | Stale `v` after a gradient spike | Lower `β2` to around 0.95 |
| NaN in fp16 | `ε=1e-8` underflows in fp16 precision | Use `ε ≥ 1e-6` |
| Huge steps on sparse gradients | `ε` too small relative to the gradient scale | Raise `ε` |

Adam can fail to converge on some deliberately constructed problems (Reddi et al., 2018), and `amsgrad=True` fixes that specific failure mode, though it rarely comes up outside of contrived counterexamples built to demonstrate the issue. Adam also isn't a substitute for a learning rate schedule, warmup followed by cosine decay still matters even with an adaptive optimizer underneath it.

## Rules of Thumb

- Start with AdamW at `lr=1e-3` for small networks trained from scratch, or `1e-4` to `2e-5` when fine-tuning a pretrained model.
- Tune `lr` before swapping optimizers. It matters more than which optimizer is chosen.
- Reach for AdaGrad only when sparsity is the defining property of the data, not as a general-purpose default.
- Reach for RMSprop mainly for RL, RNNs, or GANs, the cases above where momentum tends to hurt rather than help.
- If SGD+momentum with a well-tuned schedule is an option for a CNN, it's worth trying alongside Adam. It may generalize better.