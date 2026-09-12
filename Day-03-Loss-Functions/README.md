# Day 3 : Loss Functions in Deep Learning

## Regression Losses

### Mean Squared Error (MSE)

**Equation:**

```
MSE = (1/n) * Σ(y - ŷ)²
```

Squares every error before averaging. Large errors get punished hard because squaring amplifies them. A single outlier can dominate the total loss and pull the model toward fitting that one point instead of the overall trend.

**Gradient behavior:** the gradient of MSE scales with the size of the error. A large error produces a large gradient, which corrects fast. A small error produces a small gradient, which lets the model settle smoothly near the minimum. This is why MSE trains smoothly but breaks down when outliers are present.

**Use when:** the data has few or no outliers, and you want the model to converge cleanly.

---

### Mean Absolute Error (MAE)

**Equation:**

```
MAE = (1/n) * Σ|y - ŷ|
```

Takes the absolute value of every error instead of squaring it. Every error contributes linearly to the loss, so one extreme outlier doesn't dominate the total the way it does with MSE.

**Gradient behavior:** the gradient is constant (+1 or -1) regardless of how large the error is. This is exactly what makes MAE robust to outliers, but it comes at a cost: the gradient doesn't shrink as the model approaches the minimum, so training can oscillate near convergence instead of settling smoothly.

**Use when:** the data has outliers you don't want to dominate training, and some oscillation near the minimum is an acceptable tradeoff.

---

### Huber Loss

**Equation:**

```
Huber(y, ŷ) = 0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
            = δ * |y - ŷ| - 0.5 * δ²   if |y - ŷ| > δ
```

Quadratic for small errors, linear for large errors. The switch happens at the threshold δ, a hyperparameter you tune.

**Why it exists:** it combines MSE's smooth convergence for normal-sized errors with MAE's resistance to being dragged around by extreme ones. Small errors get the smooth quadratic treatment; large errors get capped to a linear penalty instead of being squared into dominance.

**Use when:** the data is mostly well-behaved but has a mix of a few extreme values. Housing price prediction is the standard example: most homes fall in a reasonable price band, but a handful of mansions or distressed sales would blow up an MSE loss if included untreated.

---

## Classification Losses

### Binary Cross-Entropy

Used for two-class problems (fraud vs. not fraud, churn vs. not churn). The label is a single probability, either 0 or 1.

### Categorical Cross-Entropy

Used for three or more classes. Labels are one-hot encoded: `[0, 1, 0]` for class 2 out of 3.

### Sparse Categorical Cross-Entropy

Used for three or more classes, same as categorical cross-entropy. The difference is the label format: integers instead of one-hot vectors. Class 2 is just `2`, not `[0, 1, 0]`.

**Key point:** categorical and sparse categorical cross-entropy are mathematically identical. Sparse exists so you don't have to one-hot encode your labels yourself, which also saves memory when the number of classes is large. A 1000-class problem (ImageNet, for example) means a 1000-dimensional one-hot vector per label if you use categorical. Sparse skips that entirely.

| Loss | Use case | Label format |
|---|---|---|
| Binary Cross-Entropy | 2 classes | Single probability (0 or 1) |
| Categorical Cross-Entropy | 3+ classes | One-hot encoded, e.g. `[0,1,0]` |
| Sparse Categorical Cross-Entropy | 3+ classes | Integer label, e.g. `2` |

---

## Why This Matters

A fraud detection model at a fintech uses binary cross-entropy: fraud or not fraud, one probability out. A product categorization model at an e-commerce company with hundreds of categories uses sparse categorical cross-entropy, purely to avoid one-hot encoding a label vector with hundreds of dimensions for every product.

Choosing the wrong regression loss has a real cost too. A model trained with MSE on housing data full of outlier mansion sales will skew its predictions toward fitting those mansions at the expense of accuracy on typical homes. Huber loss exists specifically to prevent that.
