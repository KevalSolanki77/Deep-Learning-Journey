# Day 13 - Batch Normalization

## The Problem

Every layer's input distribution shifts as the layers before it keep updating their weights during training. A layer that was well-calibrated for its input a few steps ago finds that input drifting under it a few steps later. Activations drift into the saturated regions of sigmoid/tanh (where the gradient is near zero) or into the dead region of ReLU (where the gradient is exactly zero), so training needs small learning rates and careful initialization just to stay stable. Batch Normalization attacks this directly: normalize each layer's input so its distribution stays consistent, batch after batch, regardless of what the earlier layers are doing.

## How It Works

For each feature, computed per mini-batch (per channel, for a CNN):

**Step 1: Batch statistics.**

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} z_i \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (z_i - \mu_B)^2$$

**Step 2: Normalize.**

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

This forces mean 0 and variance 1 across the batch, `ε` is a small constant (Keras default `1e-3`) that keeps the division from blowing up if `σ_B²` lands near zero.

**Step 3: Scale and shift.**

$$y_i = \gamma \, \hat{z}_i + \beta$$

`γ` and `β` are learned parameters, one pair per feature, trained by backpropagation exactly like any weight.

**Why `γ` and `β` exist at all.** A hard mean-0, variance-1 constraint would cap what each layer can represent. Sigmoid is nearly linear in a narrow band around 0, so clamping every activation into that band would stop the network from ever using sigmoid's nonlinear regions. `γ` and `β` give the network an escape hatch: it can learn to undo the normalization (`γ = σ_B`, `β = μ_B` recovers the original `z`) wherever that's the better option, and more generally gives it direct control over each feature's mean and spread instead of leaving that to whatever the raw weights happen to produce.

## Placement: Before or After the Activation

**Method 1 (standard): `Z → BatchNorm → activation`.** `Z`, the pre-activation, is roughly bell-shaped, close to the Gaussian BatchNorm's math assumes, so normalizing it behaves as intended, and the activation function then receives well-centered input instead of whatever scale the raw weights produced. Under this placement, the layer's bias term `b` becomes redundant: BatchNorm already subtracts the batch mean, and `β` already supplies a learned shift, so a separate `b` is two parameters doing the same job. Standard practice is `Dense(..., use_bias=False)` for any layer immediately followed by BatchNorm.

**Method 2: `Z → activation → BatchNorm`.** Valid, used in practice, but empirical rather than principled. By the time BatchNorm sees the values, the activation function has already saturated or clipped them (ReLU has already zeroed out everything negative), so BatchNorm is normalizing an already-distorted distribution instead of preventing the distortion in the first place.

## Training vs. Inference

- **Training**: normalizes using the current mini-batch's own mean and variance, and simultaneously updates a running average of both statistics via an exponential moving average, Keras's default momentum is 0.99.
- **Inference**: normalizes using the frozen running averages collected during training, not the statistics of whatever batch happens to be running through the model. This is what makes inference deterministic. A prediction for one input doesn't depend on which other inputs happened to be batched alongside it.

**Keras-specific detail worth getting right.** `model.eval()` is PyTorch, not Keras, there's no `model.eval()` call here. In Keras the switch is the `training` argument: `model.predict()`, `model.evaluate()`, and direct inference calls automatically run with `training=False`, so BatchNorm layers use the frozen running statistics as intended. The failure mode described, inconsistent predictions because batch statistics are still active, happens in Keras if you call the model directly with `training=True` (or omit the argument in a custom training loop where it's needed) during what should be an inference pass. Same underlying bug as forgetting PyTorch's `model.eval()`, different mechanism for triggering it.

## Why It Helps

- **Smoother loss landscape.** Santurkar et al. (2018) showed the long-standing "reduces internal covariate shift" explanation doesn't hold up as the actual mechanism; what BatchNorm measurably does is smooth the loss surface and make gradients more predictable step to step, which is what actually lets training proceed faster and more stably.
- **Higher learning rates, faster convergence.** A smoother loss surface tolerates bigger steps without diverging.
- **Less sensitivity to initialization.** Day 12's zero/constant/small-random failure modes are all less damaging when BatchNorm is re-centering every layer's input regardless of how the weights started.
- **Mild regularization**, as a side effect, not the primary purpose. Each mini-batch has slightly different statistics, so the normalization a given example receives varies run to run, adding a small amount of training noise similar in spirit to Dropout (Day 9).

## Where It Breaks

- **Small batch sizes.** Batch statistics computed from a handful of examples are noisy, which undermines the entire premise of normalizing against them. GroupNorm or InstanceNorm (which don't depend on batch size) replace BatchNorm in this regime.
- **Sequence models / RNNs.** Variable-length sequences make a consistent per-batch statistic awkward to define across time steps.
- **Train/inference distribution mismatch.** If the running statistics collected during training don't represent the data seen at inference, (a classic cause: training and deployment data drift apart over time), the frozen averages become stale and predictions degrade.
- **Transformers use LayerNorm instead.** LayerNorm normalizes across the features of one sample, with no dependency on other samples in the batch at all, which sidesteps the small-batch and sequence-length problems above entirely.

## Real Cases

- **ResNet, YOLO, EfficientNet**: `Conv → BatchNorm → ReLU` is the standard block repeated throughout the architecture.
- **Medical imaging**, where batch sizes often run 1 to 4 due to large image sizes and memory limits, is exactly the small-batch regime BatchNorm breaks in, GroupNorm is the standard swap.
- **Serving bugs**: a model that scores well offline but behaves erratically in production is a common symptom of a BatchNorm mode bug (running in training mode at inference) or a drift issue (running statistics no longer matching live traffic).

## The Experiment

**Without BatchNorm:**

```python
model1 = Sequential([
    Input(shape=(2,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**With BatchNorm:**

```python
model2 = Sequential([
    Input(shape=(2,)),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(4, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
```

Neither `Dense` layer sets `use_bias=False` here, despite both being followed immediately by `BatchNormalization()`. Per the placement section above, that bias term is redundant weight the network has to learn and immediately cancel out, worth fixing in the next pass at this notebook.

```python
model2.summary()
```

| | Params |
|---|---|
| Trainable (Dense weights + BN γ, β) | 801 |
| Non-trainable (BN running mean, running variance) | 104 |
| Total | 2,509 (incl. optimizer state) |

The 104 non-trainable parameters are exactly the running statistics: 2 numbers (mean, variance) per feature, across the 32 + 16 + 4 = 52 total units the three `BatchNormalization` layers sit on, `2 × 52 = 104`. The 104 trainable BN parameters (`γ`, `β` pairs, same count) are learned by gradient descent alongside the Dense weights, confirming the note above: BatchNorm adds both a trainable cost and a non-trainable bookkeeping cost, not just one or the other.

![Comparison Between With & Without Batch Normalization](https://i.ibb.co/xqYKBR9k/batch-norm.png)

Reading the accuracy curves: the BatchNorm run starts lower (around 0.52 at epoch 0 versus 0.82 without it) but climbs past 90% accuracy within the first 10 epochs. The no-BatchNorm run doesn't cross 90% until roughly epoch 30. Both curves converge to the same final accuracy, around 0.96, by epoch 60 and hold there for the remaining epochs. This matches the note directly: BatchNorm here is a training-speed effect, reaching a given accuracy level in far fewer epochs, not a final-accuracy effect, since both models land in the same place eventually.