# Day 10 - Regularization 

## Why Neural Networks Overfit

A network with many parameters has the capacity to draw a decision boundary that threads through every training point exactly, noise included. More parameters means more capacity to memorize rather than generalize, and memorized noise doesn't show up in test data, so performance there suffers even as training performance looks perfect.

## Three Ways to Fight It

- **More data**: real data or augmented synthetic examples give the network less room to memorize, since a memorized pattern that only fits some of the data stops looking like a shortcut.
- **Architectural changes**: dropout (Day 9) and early stopping (Day 8) both constrain how the network trains without touching the cost function itself.
- **Regularization terms**: add a penalty to the cost function that discourages large weights directly. This is what L1 and L2 regularization do, and what this entry covers.

## The Math

The plain cost function averages the loss `L` over `m` training examples:

$$J(w) = \frac{1}{m}\sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)$$

**L2 (Ridge )regularization** adds the squared magnitude of every weight, scaled by `λ`:

$$J(w) = \frac{1}{m}\sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right) + \frac{\lambda}{2m}\sum_{j} w_j^2$$

`λ` controls the trade-off: `λ = 0` recovers the plain cost function, and a larger `λ` pushes harder against large weights, even at the cost of fitting the training data less well. Biases are left out of the sum, only the weights get penalized.

Differentiating the L2 term and folding it into the gradient descent update rule:

$$w := w - \eta\left(\frac{\partial L}{\partial w} + \frac{\lambda}{m}w\right) = w\left(1 - \frac{\eta\lambda}{m}\right) - \eta\frac{\partial L}{\partial w}$$

Every update multiplies `w` by a factor just under 1 before the usual gradient step runs. That's the entire mechanism behind the name **weight decay**: each step shrinks every weight toward zero by a fixed proportion of its current value, on top of whatever the loss gradient does.

**L1 (Lasso) regularization** adds the absolute value of the weights instead of the square:

$$J(w) = \frac{1}{m}\sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right) + \frac{\lambda}{m}\sum_{j} |w_j|$$

The derivative of `|w|` is `sign(w)`, a constant `+1` or `-1` regardless of how large `w` is, so the update rule looks different in one key way:

$$w := w - \eta\left(\frac{\partial L}{\partial w} + \frac{\lambda}{m}\,\text{sign}(w)\right)$$

## Why L1 Zeros Weights and L2 Doesn't

This difference in the update rule is the entire reason for the sparsity behavior your notes describe. L2's decay term is proportional to `w`: as `w` shrinks toward zero, the pull toward zero shrinks with it, so the weight approaches zero but the pull weakens exactly as fast, and it never quite arrives. L1's pull is a constant amount every step, independent of how small `w` already is. Once a weight gets close enough to zero that a full step would carry it past zero, it lands at exactly zero and stays there. Same idea in one line: L2 shrinks weights proportionally forever, L1 subtracts a fixed amount until there's nothing left to subtract.

## The Experiment

Same architecture, same `make_moons` data, same optimizer, only the `kernel_regularizer` on the two hidden `Dense` layers changes. All three train for 2000 epochs with `Adam(learning_rate=0.01)`.

**Baseline, no regularization:**

```python
model1 = Sequential()
model1.add(Input(shape = (2,)))
model1.add(Dense(128, activation="relu"))
model1.add(Dense(128, activation="relu"))
model1.add(Dense(1, activation='sigmoid'))

adam = Adam(learning_rate=0.01)
model1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history1 = model1.fit(X, y, epochs=2000, validation_split = 0.2, verbose=0)
```

```python
model1.get_weights()[0].reshape(256).min(), model1.get_weights()[0].reshape(256).max()
# (-4.931019, 3.516342)
```

**L1, λ = 0.001:**

```python
model2 = Sequential()
model2.add(Input(shape = (2,)))
model2.add(Dense(128, activation="relu", kernel_regularizer=l1(0.001)))
model2.add(Dense(128, activation="relu", kernel_regularizer=l1(0.001)))
model2.add(Dense(1, activation='sigmoid'))

adam = Adam(learning_rate=0.01)
model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history2 = model2.fit(X, y, epochs=2000, validation_split = 0.2, verbose=0)
```

```python
model2.get_weights()[0].reshape(256).min(), model2.get_weights()[0].reshape(256).max()
# (-0.21197155, 0.21478392)
```

**L2, λ = 0.001:**

```python
model3 = Sequential()
model3.add(Input(shape = (2,)))
model3.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
model3.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
model3.add(Dense(1, activation='sigmoid'))

adam = Adam(learning_rate=0.01)
model3.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history3 = model3.fit(X, y, epochs=2000, validation_split = 0.2, verbose=0)
```

```python
model3.get_weights()[0].reshape(256).min(), model3.get_weights()[0].reshape(256).max()
# (-1.5525979, 0.8184027)
```

## Reading the Results

| Model | Weight range | Spread |
|---|---|---|
| No regularization | -4.93 to 3.52 | 8.45 |
| L1 (λ = 0.001) | -0.21 to 0.21 | 0.43 |
| L2 (λ = 0.001) | -1.55 to 0.82 | 2.37 |

The unregularized first-layer weights spread across a range of 8.45. The same `λ` value compresses that range to 2.37 under L2 and all the way to 0.43 under L1, nearly 20 times tighter than the baseline. This lines up directly with the math above: L1's constant per-step pull toward zero is far more aggressive at this `λ` than L2's proportional shrink, which is also why L1 is the one that tends to produce weights sitting at or near exactly zero rather than merely small.

The notebook plots decision boundaries and loss curves for all three models rather than printing final loss or accuracy numbers, so there's no train/test score to report the way earlier entries had. The weight compression above is the measurable result this run actually produced.