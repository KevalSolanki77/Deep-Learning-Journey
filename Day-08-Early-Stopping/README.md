# Day 8 (Deep Learning) - Early Stopping

## The Problem: Overfitting From Too Many Epochs

Train long enough and the model starts memorizing the training set. Training loss keeps falling while loss on unseen data flattens or climbs, so each extra epoch fits the training points tighter and predicts new points worse. A fixed epoch count (say 200) ignores this. The right count depends on the data, the architecture and the learning rate, and nobody knows it before training.

## The Mechanism

Early stopping replaces the fixed epoch count with a rule. After each epoch, the callback evaluates the monitored metric on a validation set, tracks the best value so far, and counts the epochs since that best value. When the count reaches `patience`, training halts.

For a metric that should decrease (`mode="min"`), with best value $L_{best}$ and `min_delta` $\delta$:

$$\text{improved at epoch } t \iff L_{val}(t) < L_{best} - \delta$$

On improvement, $L_{best} \leftarrow L_{val}(t)$ and the wait counter resets to 0. Otherwise the counter goes up by 1. Training stops when the counter equals `patience`.

## Implementation in Keras

Keras exposes this as a callback. You build an `EarlyStopping` object and pass it to `fit(callbacks=...)`.

| Parameter | Meaning | Value in this run |
|---|---|---|
| `monitor` | Metric to track | `val_loss` |
| `patience` | Epochs without improvement to wait before stopping | 20 |
| `min_delta` | Smallest change that counts as an improvement | 1e-7 |
| `mode` | `min` stops when the metric stops decreasing, `max` when it stops increasing, `auto` infers from the metric name | `auto` |
| `baseline` | Training also stops if the metric fails to beat this value, `None` disables the check | `None` |
| `restore_best_weights` | `True` rolls the model back to the best epoch's weights on stop, `False` keeps the last epoch's weights | `False` |
| `verbose` | Prints a message when training stops | 1 |

`auto` picks `min` for `val_loss` and `max` for accuracy metrics. With `min_delta` at 1e-7, any measurable decrease counts as an improvement.

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
```

```python
X, y = make_circles(n_samples=100, noise=0.1, random_state=2)
sns.scatterplot(x = X[:,0],y = X[:,1],hue=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
```

100 points on two concentric circles, split into 80 training and 20 validation points. `X_test` doubles as `validation_data`, so the same 20 points decide when training stops and also supply the final metrics reported below. No separate held-out test set exists, which makes those final numbers optimistic.

## Baseline: 200 Epochs Without Early Stopping

```python
model = Sequential()

model.add(Input(shape = (2,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=1)
```

```python
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
```

| Epoch | Train loss | Val loss | Val accuracy |
|---|---|---|---|
| 1 | 0.6921 | 0.6935 | 0.45 |
| 20 | 0.6749 | 0.7296 | 0.40 |
| 50 | 0.6595 | 0.7479 | 0.35 |
| 72 | 0.6439 | 0.7492 | 0.20 |
| 100 | 0.6191 | 0.7253 | 0.30 |
| 150 | 0.5699 | 0.6905 | 0.50 |
| 200 | 0.5251 | 0.6577 | 0.60 |

Training loss falls the whole way, from 0.6921 to 0.5251. Validation loss takes a different path: it climbs from 0.6935 to a peak of 0.7492 at epoch 72, then recovers, drops below its epoch-1 value at epoch 149, and ends at 0.6577. Under the callback's rules, the longest stretch without a new best in this run lasts 147 epochs (epoch 2 through epoch 148).

## With Early Stopping

```python
model = Sequential()

model.add(Input(shape = (2,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

```python
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0000001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
```

```python
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, callbacks=callback)
```

```python
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
```

```text
Epoch 21: early stopping
```

Epoch 1 set the best `val_loss` (0.7005), and no later epoch beat it. The wait counter rose by 1 each epoch from epoch 2 and reached 20 at epoch 21, so training halted there.

| | Epochs run | Train loss | Val loss | Val accuracy |
|---|---|---|---|---|
| No early stopping | 200 | 0.5251 | 0.6577 | 0.60 |
| Early stopping, patience 20 | 21 | 0.6767 | 0.7382 | 0.35 |

## Reading the Result

The callback did its job: `val_loss` failed to improve for 20 epochs, so training stopped. The stopped model still scores worse than the one trained for 200 epochs, and the baseline curve shows why. At epoch 21 the training accuracy sat at 0.61, so the model was underfit and had not started memorizing anything. The rise in validation loss came from a slow start, and the baseline run recovered from it on its own. A patience of 20 cut training off during that dip: validation loss peaked at epoch 72 and got back below its starting value at epoch 149, 128 epochs after the stop.

Two factors made the dip look like a stopping signal:

- **Small validation set.** With 20 points, one sample moves validation accuracy by 5 points, and validation loss swings with it.
- **Patience shorter than the dip.** The baseline run needed `patience` of 148 or more to survive its 147-epoch stretch without a new best.

`restore_best_weights=False` also shaped the outcome. The stopped model holds the epoch-21 weights, 20 epochs past the best epoch. Setting it to `True` would have returned the epoch-1 weights, one epoch away from random initialization.

Early stopping fits cases where validation loss turns upward after training loss has already converged, the classic overfitting shape. Here, the network was still learning the circular boundary when the callback fired.