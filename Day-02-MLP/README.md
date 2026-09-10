# Customer Churn Prediction — First Neural Network (MLP)

## What this project does

A Multi-Layer Perceptron predicts whether a customer will churn, using the Customer Churn dataset. This is the first hands-on build after perceptron theory: one hidden layer, forward propagation, and a binary output.

---

## Data Pipeline

1. Load `customer_churn_dataset-training-master.csv`.
2. Drop rows with missing values.
3. Drop `CustomerID`. It identifies a row; it doesn't describe a customer.
4. Encode `Subscription Type` with `OrdinalEncoder`, using the order Basic < Standard < Premium. This is a genuine ordinal variable, so rank order carries information a plain one-hot encoding would throw away.
5. One-hot encode `Contract Length` and `Gender` with `pd.get_dummies`, dropping the first category to avoid the dummy variable trap.
6. Split into train and validation sets (70/30).
7. Scale features with `StandardScaler`, fit on train only, then apply to validation. Fitting the scaler on validation data would leak information from the set you're supposed to be testing against.

---

## Architecture

```
Input layer:  11 features
Hidden layer: 3 neurons, sigmoid activation
Output layer: 1 neuron, sigmoid activation
```

In MLP notation: `a[0]` is the 11-dimensional input vector. Layer 1 computes `z[1] = W[1]·a[0] + b[1]`, then `a[1] = sigmoid(z[1])`, producing 3 activations. Layer 2 computes `z[2] = W[2]·a[1] + b[2]`, then `a[2] = sigmoid(z[2])`, producing the final churn probability.

Three neurons in the hidden layer keeps the model small on purpose. Eleven input features and a binary target don't need a wide network to find a decent decision boundary. The goal here is watching forward propagation work end to end, not squeezing out the best accuracy.

---

## Forward Propagation, in Practice

Each training example moves through the network in one direction: input to hidden layer, hidden layer to output. At every layer, the network computes a weighted sum of the inputs it receives, adds a bias, and squashes the result through sigmoid to keep values between 0 and 1.

The output layer's sigmoid produces a probability, not a hard label. `model.predict()` returns that probability. A threshold turns it into a class:

```python
y_hat = np.where(y_prob > threshold, 1, 0)
```

The threshold here is 0.8, not the default 0.5. Raising the threshold makes the model demand more confidence before it predicts churn. That choice has a real cost behind it: a churn model feeding a retention campaign trades off false positives (a retention offer wasted on a customer who wasn't leaving) against false negatives (a customer lost that the model should have flagged).

---

## Training Setup

- **Loss:** Binary Crossentropy. Matches the sigmoid output and binary target.
- **Optimizer:** Adam. Adaptive learning rate, a standard default for a first model.
- **Epochs:** 10.

---

## Why This Matters

Subscription businesses (telecom, SaaS, streaming) run churn models constantly to trigger retention offers before a customer cancels. A 3-neuron MLP won't beat XGBoost on tabular data like this in most real deployments, but building it by hand shows the forward pass with nothing hidden: probability in, weighted sum, activation, probability out.

This also sets up backpropagation. The notation here (`W[l]`, `b[l]`, `z[l]`, `a[l]`) is the same notation backprop uses to compute gradients layer by layer.

---

## Next Steps

- Inspect `model.layers[0].get_weights()` and connect the shapes back to the architecture. Eleven inputs into 3 neurons means a weight matrix of shape `(11, 3)`.
- Compare this MLP against a tree-based baseline (Random Forest or XGBoost) on the same preprocessed data. Tabular data usually favors gradient boosting over small neural nets, and this is worth confirming rather than assuming.
- Move to backpropagation: derive how the gradient of the loss flows backward through `a[2] → z[2] → a[1] → z[1]` to update `W[1]`, `W[2]`, `b[1]`, `b[2]`.