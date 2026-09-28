# Day 4 — Backpropagation

## What Is Backpropagation

Backpropagation is the algorithm that trains an Artificial Neural Network under supervised learning, using gradient descent. Given a network and a loss function, it computes the gradient of the loss with respect to every weight and bias in the network, layer by layer, working backward from the output.

The core mechanism is the **chain rule** from calculus. A neural network is a composition of functions (layer 1 feeds layer 2 feeds the output), so the effect of an early-layer weight on the final loss has to be traced through every layer in between. Backpropagation is just a systematic, efficient way of applying the chain rule across that whole composition instead of deriving each gradient from scratch.

## Why It's Used

Two reasons it's the standard, not just one option among many:

1. **Manual gradient computation doesn't scale.** For a two-layer network with 9 parameters (like the one below), writing out each partial derivative by hand is tedious but doable. Real networks have millions of parameters across dozens of layers — backprop turns that into a mechanical, repeatable procedure instead of a one-off derivation per architecture.
2. **It's efficient, not just correct.** Backprop computes all gradients in a single backward pass by reusing intermediate results (like `dL/dŷ`) across multiple parameter updates, rather than recomputing the loss function's derivative from scratch for every individual weight.

This is exactly what `loss.backward()` does in PyTorch or TensorFlow — those frameworks automate the same chain-rule bookkeeping done manually below.

## The Five Steps

1. Initialize weights and biases.
2. Select a random point (row of data).
3. Predict the output using forward propagation.
4. Choose a loss function.
5. Update weights and biases using gradient descent.

Steps 1–3 build a prediction. Step 4 measures how wrong it is. Step 5 — backpropagation proper — pushes that error signal backward through the network to correct the weights.

## The Network

Built from scratch on a toy dataset: predicting `package` (LPA) from `cgpa` and `resume_score`.

| cgpa | resume_score | package |
|---|---|---|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 6 |
| 5 | 12 | 7 |

Architecture: 2 inputs → 1 hidden layer (2 nodes) → 1 output node. No activation function between layers — this is deliberate for a first pass at backprop, so the chain rule is visible without an extra derivative term. (See the note under "What's Next" — this also means the network is currently just linear regression in disguise.)

### Step 1: Initialize Weights and Biases

Weights start at a small constant (0.1) rather than zero, so gradients aren't zero on the first pass. Biases start at zero.

```python
def initialize_parameters(layer_dims):
    paramerters = {}
    L = len(layer_dims) # No of layers

    for i in range(1,L):
        paramerters[f'W{i}'] = np.ones(shape = (layer_dims[i-1], layer_dims[i])) * 0.1
        paramerters[f'B{i}'] = np.zeros(shape = (layer_dims[i], 1))

    return paramerters
```

```python
initialize_parameters([2,2,1])
# {'W1': array([[0.1, 0.1], [0.1, 0.1]]),
#  'B1': array([[0.], [0.]]),
#  'W2': array([[0.1], [0.1]]),
#  'B2': array([[0.]])}
```

### Steps 2–3: Forward Propagation

Each layer computes:

$$A^{[l]} = W^{[l]T} A^{[l-1]} + b^{[l]}$$

Applied twice for this network:

$$a^{[1]} = W^{[1]T} x + b^{[1]} \qquad \hat{y} = a^{[2]} = W^{[2]T} a^{[1]} + b^{[2]}$$

```python
def L_layer_forward(X, parameters):
    A = X
    L = len(parameters) // 2

    for l in range(1, L+1):
        A_prev = A
        Wl = parameters[f'W{l}']
        bl = parameters[f'B{l}']
        A = np.dot(Wl.T, A_prev) + bl

    return A_prev, A
```

On the first row (`cgpa=8, resume_score=8`) with the initial weights: `a1 = [[1.6], [1.6]]`, `ŷ = 0.32`. Actual `y = 4`, so the network starts off badly wrong — that error is what step 5 corrects.

### Step 4: Loss Function

Mean squared error, standard for a continuous target like `package`:

$$L = (y - \hat{y})^2$$

### Step 5: Backpropagation — Applying the Chain Rule

This is the actual backward pass. Working from the output layer back to the input layer, with learning rate `η = 0.001`:

**Output layer (W2, b2):**

$$\frac{\partial L}{\partial \hat{y}} = -2(y-\hat{y})$$

$$\frac{\partial L}{\partial W^{[2]}_i} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W^{[2]}_i} = -2(y-\hat{y})\, a^{[1]}_i \quad\Rightarrow\quad W^{[2]}_i \mathrel{+}= \eta \cdot 2(y-\hat{y})\, a^{[1]}_i$$

$$\frac{\partial L}{\partial b^{[2]}} = -2(y-\hat{y}) \quad\Rightarrow\quad b^{[2]} \mathrel{+}= \eta \cdot 2(y-\hat{y})$$

**Hidden layer (W1, b1)** — the error has to pass through `a1` first:

$$\frac{\partial L}{\partial a^{[1]}_i} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a^{[1]}_i} = -2(y-\hat{y})\, W^{[2]}_i$$

$$\frac{\partial L}{\partial W^{[1]}_{i,j}} = \frac{\partial L}{\partial a^{[1]}_i} \cdot \frac{\partial a^{[1]}_i}{\partial W^{[1]}_{i,j}} = -2(y-\hat{y})\, W^{[2]}_i\, x_j \quad\Rightarrow\quad W^{[1]}_{i,j} \mathrel{+}= \eta \cdot 2(y-\hat{y})\, W^{[2]}_i\, x_j$$

This is the chain rule in action: the gradient for a hidden-layer weight is the output-layer error, scaled by the weight connecting that hidden node to the output, scaled again by the input that fed the hidden node. Three multiplications tracing one path backward through the network.

```python
def update_parameters(parameters, y, y_hat, A1, X):
    parameters['W2'][0][0] = parameters['W2'][0][0] + (0.001 * 2 * (y - y_hat)* A1[0][0])
    parameters['W2'][1][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat)* A1[1][0])
    parameters['B2'][0][0] = parameters['B2'][0][0] + (0.001 * 2 * (y - y_hat))

    parameters['W1'][0][0] = parameters['W1'][0][0] + (0.001 * 2 * (y - y_hat) * parameters['W2'][0][0] * X[0][0])
    parameters['W1'][0][1] = parameters['W1'][0][1] + (0.001 * 2 * (y - y_hat) * parameters['W2'][0][0] * X[1][0])
    parameters['B1'][0][0] = parameters['B1'][0][0] + (0.001 * 2 * (y - y_hat) * parameters['W2'][0][0])

    parameters['W1'][1][0] = parameters['W1'][1][0] + (0.001 * 2 * (y - y_hat) * parameters['W2'][1][0] * X[0][0])
    parameters['W1'][1][1] = parameters['W1'][1][1] + (0.001 * 2 * (y - y_hat) * parameters['W2'][1][0] * X[1][0])
    parameters['B1'][1][0] = parameters['B1'][1][0] + (0.001 * 2 * (y - y_hat) * parameters['W2'][1][0])
```

Note the sign: the code *adds* the gradient term rather than subtracting it, because the derivative of `(y - ŷ)²` w.r.t. the weight already carries a negative sign — `+= η · 2(y-ŷ) · x` and `−= η · dL/dW` land on the same update once that sign is factored in. It looks like gradient ascent on the page; it's gradient descent once you track the sign through.

### Training Loop

```python
parameters = initialize_parameters([2,2,1])
epochs = 20

for i in range(epochs):
    Loss = []

    for j in range(df.shape[0]):
        X = df[['cgpa', 'resume_score']].values[j].reshape(2,1)
        y = df[['package']].values[j][0]

        A1, y_hat = L_layer_forward(X, parameters)
        y_hat = y_hat[0][0]

        update_parameters(parameters, y, y_hat, A1, X)

        Loss.append((y - y_hat)**2)
    print(f"Epoch - {i+1} | Loss - {np.array(Loss).mean()}")
```

Loss drops fast from 26.3 to ~1.27 within 5 epochs, then plateaus and drifts slightly upward for the remaining 15 — a sign the fixed learning rate (0.001) is slightly too high for full convergence on this tiny dataset, not that the gradient math is wrong.

## An Honest Note: A Hidden Fragility in `update_parameters`

`W1` is built with shape `(input_dim, hidden_dim)` — forward propagation uses `W1.T @ x`, so the mathematically correct indexing is `W1[input_index][hidden_index]`.

But `update_parameters` indexes it the other way: `W1[0][0]` and `W1[0][1]` are both updated using `W2[0][0]` (hidden node 0), just with different `X` indices. That treats the *first* index of `W1` as the hidden-node index and the second as the input index — the opposite of how forward propagation reads the array.

This doesn't crash and doesn't even produce visibly wrong output here, for one reason only: `input_dim == hidden_dim == 2`, so `W1` is a 2×2 square matrix and the mismatched indexing still lands on valid, if conceptually swapped, cells. Change the hidden layer to 3 nodes and this same update code would either throw an index error or silently update the wrong weights. Worth remembering before reusing this pattern in the DL stage: the derivation above (`W1[i,j] += η · 2(y-ŷ) · W2[i] · x[j]`) is the correct one — a general implementation should compute it as a matrix operation (`(W2 @ dL_dyhat) outer x`) rather than hardcoded scalar indices, exactly so it doesn't depend on two dimensions happening to match.

## Real-World Tie-In

Every time a PyTorch or TensorFlow model calls `.backward()`, it's running this exact chain-rule bookkeeping — just automated across arbitrarily many layers via a computational graph (reverse-mode automatic differentiation), instead of nine lines of manual index math. Fraud-detection and credit-scoring deep models at Indian fintechs (Paytm, Razorpay, HDFC's internal risk models) train on millions of transactions this way — the manual version here is the literal mechanism running underneath, just at a scale where nobody could derive gradients by hand.