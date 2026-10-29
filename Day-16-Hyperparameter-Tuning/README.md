# Day 16 - Hyperparameter Optimization with Keras Tuner

## The Shared Idea

Gradient descent optimizes internal parameters (weights and biases) by following the loss surface. It cannot optimize hyperparameters—structural choices like depth, layer widths, learning rates, activation functions, regularization rates, or batch sizes. These must be declared before training begins.

Manual tuning degenerates into uncalibrated trial-and-error. Keras Tuner formalizes hyperparameter exploration: you parameterize the network definition inside a `build_model(hp)` function or a custom `kt.HyperModel` class, replace fixed constants with bounded ranges, and let an optimization algorithm evaluate candidate configurations against a held-out validation metric.

| Tuner | Search Mechanism | Resource Efficiency | Recommended Use Case |
|---|---|---|---|
| **RandomSearch** | Samples hyperparameter values uniformly at random | Baseline; inefficient over high dimensions | Default baseline; wide, unconstrained search spaces |
| **BayesianOptimization** | Uses Gaussian processes or tree-structured Parzen estimators to build a surrogate model of the objective function | High; spends compute where performance gains are probabilistic | Expensive training runs (large models or slow datasets) |
| **Hyperband** | Multi-fidelity bandit search via successive halving; trains many models for few epochs, discarding bottom performers | Highest; minimizes wasted compute on dead configurations | Standard first choice for deep architectures with early stopping |
| **GridSearch** | Exhaustive cartesian product evaluation | Extremely low; factorial growth | Tiny spaces only (e.g., final micro-sweep over 2–3 values) |

---

## Defining the Search Space

Keras Tuner exposes the `hp` (`HyperParameters`) object to define candidate dimensions:

```python
# Integer bounds: layer depth, layer width, kernel size
hp.Int('units', min_value=32, max_value=512, step=32)

# Floating point: dropout rates, weight decay, learning rates
hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.05)

# Logarithmic scaling for multiplicative parameters (e.g., learning rates)
hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')

# Categorical choices: activation functions, optimizer types
hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])

# Binary toggle: architectural ablation, inclusion of BatchNorm or Dropout
hp.Boolean('use_batch_norm')
```

> **Why logarithmic sampling matters:** Learning rates operate on multiplicative scales ($10^{-4}$ to $10^{-3}$ represents the same functional step as $10^{-3}$ to $10^{-2}$). A uniform linear sample between $0.0001$ and $0.1$ places 90% of all trials above $0.01$, starving the lower, often optimal regimes.

---

## Step-by-Step Code Implementations

The following examples reproduce the progression from isolated component tuning to end-to-end joint search, using a 23-dimensional feature space derived from tabular student telemetry.

### 1. Tuning the Optimizer Choice

Holding architecture constant to isolate the optimizer's performance under default initialization:

```python
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, Dense

def build_optimizer_model(hp):
    model = Sequential([
        Input(shape=(23,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    
    optimizer_name = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])
    model.compile(
        optimizer=optimizer_name,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner_opt = kt.RandomSearch(
    build_optimizer_model,
    objective='val_accuracy',
    max_trials=10,
    project_name='tuning_optimizer',
    overwrite=True
)

tuner_opt.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
best_optimizer = tuner_opt.get_best_hyperparameters()[0].values
```

---

### 2. Tuning Layer Width (Neuron Count)

Testing neuron density over a continuous step interval:

```python
def build_width_model(hp):
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    model = Sequential([
        Input(shape=(23,)),
        Dense(units=units, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_width = kt.RandomSearch(
    build_width_model,
    objective='val_accuracy',
    max_trials=20,
    project_name='tuning_layer_width',
    overwrite=True
)

tuner_width.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
best_width = tuner_width.get_best_hyperparameters()[0].values
```

---

### 3. Tuning Architectural Depth (Number of Hidden Layers)

Dynamically unrolling hidden layers within the graph construction loop:

```python
def build_depth_model(hp):
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    model = Sequential()
    model.add(Input(shape=(23,)))

    for _ in range(num_layers):
        model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_depth = kt.RandomSearch(
    build_depth_model,
    objective='val_accuracy',
    max_trials=5,
    project_name='tuning_layer_depth',
    overwrite=True
)

tuner_depth.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
best_depth = tuner_depth.get_best_hyperparameters()[0].values
```

---

### 4. Full Joint Search Space

Simultaneously searching over depth, layer-specific widths, regularization toggles, and optimizer candidates:

```python
from keras.layers import Dropout

def build_combined_model(hp):
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    model = Sequential()
    model.add(Input(shape=(23,)))

    for i in range(num_layers):
        units = hp.Int(f'units_{i + 1}', min_value=32, max_value=512, step=32)
        model.add(Dense(units=units, activation='relu'))
        
        if hp.Boolean('dropout'):
            model.add(Dropout(rate=0.25))

    optimizer_name = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer_name, loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_full = kt.RandomSearch(
    build_combined_model,
    objective='val_accuracy',
    max_trials=30,
    project_name='tuning_full_model',
    overwrite=True
)

tuner_full.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

### 5. Tuning Batch Size via `HyperModel.fit`

Because `batch_size` is an argument to `model.fit()` rather than `model.compile()`, sub-class `kt.HyperModel` and override the execution step:

```python
class TabularHyperModel(kt.HyperModel):
    def build(self, hp):
        units = hp.Int('units', min_value=32, max_value=256, step=32)
        model = Sequential([
            Input(shape=(23,)),
            Dense(units, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128]),
            **kwargs,
        )

batch_tuner = kt.RandomSearch(
    TabularHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    project_name='tuning_batch_size',
    overwrite=True
)

batch_tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

## Production Rules of Thumb & Best Practices

- **Impact Hierarchy:** Prioritize parameters in order of variance contribution:
  $$\text{Learning Rate} > \text{Network Architecture (Depth/Width)} > \text{Regularization (Dropout/Weight Decay)} > \text{Batch Size} > \text{Activations}$$
- **Early Termination:** Always pass `callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]` into `tuner.search()` to halt non-converging trials early.
- **Averaging Seed Variance:** Set `executions_per_trial=2` or `3` when comparing configurations with tight margins. A configuration is only as good as its stability across initializations.
- **State Invalidation:** Always pass `overwrite=True` when modifying the parameter search boundaries. Otherwise, Keras Tuner reloads the previous iteration's checkpoint history from disk.
- **Retraining Protocol:** Never deploy the raw model checkpoint emitted mid-search. Extract the winning dictionary via `tuner.get_best_hyperparameters()[0]`, reconstruct the model clean, train from scratch on combined training/validation data, and run evaluation once on the holdout test partition.

---

## Failure Modes & Diagnostics

| Symptom | Underlying Cause | Corrective Action |
|---|---|---|
| **Trials finish instantly / crash out** | OOM from unbounded width/batch combinations | Clamp maximum ranges (`max_value`) on deeper layers; monitor peak VRAM. |
| **Search results fail to improve over baseline** | Space is too broad for `max_trials` count | Narrow bounds around an initial sensible default; swap `RandomSearch` for `Hyperband`. |
| **High validation metric, terrible test score** | Information leakage or test-set tuning | Ensure scalers are fitted strictly on `X_train`. Never pass `(X_test, y_test)` as validation data to the search loop. |
| **Random fluctuations dominate leaderboard** | High seed sensitivity on single executions | Set `executions_per_trial=2` or `3` to smooth out outlier initializations. |
| **Tuning plateaus prematurely** | Learning rate step intervals are linear | Switch learning rate definition to `hp.Float(..., sampling='log')`. |

---

## Real-World Applications

| Domain | Optimization Target | Critical Hyperparameters |
|---|---|---|
| **Fraud Detection** | Area Under the PR/ROC Curve (`val_auc`) | Focal loss $\gamma$, classification threshold, class weighting |
| **Defect Detection (Vision)** | Validation Loss / F1-score | Fine-tuning learning rate ($10^{-5}$–$10^{-3}$), unfreezing depth |
| **Churn Forecasting** | Recall at Fixed Precision | Drop-out rates, class-weighted binary cross-entropy parameters |
| **Large-Scale Ads Ranking** | Log-loss / Cross-Entropy (0.001 delta = high business value) | Learning rate schedules, embedding dimension sizes, $L_2$ penalties |