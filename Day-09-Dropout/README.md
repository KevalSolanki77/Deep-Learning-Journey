# Day 9 - Dropout

## The Problem: Overfitting From High-Capacity Networks

A wide, deep network has enough parameters to memorize training data outright rather than learn the pattern behind it. Early stopping (Day 8) treats the symptom by cutting training short. Dropout treats the cause: it stops the network from building the kind of brittle, over-specific representations that overfitting produces in the first place.

## What Dropout Does

During training, each Dropout layer zeroes out a random fraction `p` of the activations flowing through it, on every forward pass. A fresh random mask gets drawn for every batch, not once per epoch, so across a single epoch's worth of batches a given neuron gets dropped and kept many times over, in different combinations each time.

```python
model_2 = Sequential()

model_2.add(Input(shape = (1,)))
model_2.add(Dense(128, activation="relu"))
model_2.add(Dropout(0.2))
model_2.add(Dense(128, activation="relu"))
model_2.add(Dropout(0.2))
model_2.add(Dense(1, activation="linear"))
```

`Dropout(0.2)` placed after a `Dense` layer drops 20% of that layer's outputs before they reach the next layer, for that batch only.

## Why It Works: Breaking Co-adaptation

Without dropout, neurons can settle into narrow, cooperative roles: neuron A only works correctly because it can lean on neuron B's specific output every time. That combination memorizes training-set quirks instead of general patterns. Removing neurons at random on each batch means no neuron can count on any specific other neuron being present, so each one has to carry information that stays useful on its own. The network ends up learning redundant, distributed representations instead of narrow dependencies.

## The Random Forest Analogy, Refined

The instinct here is right, with one distinction worth keeping straight. Random Forest trains genuinely separate trees, each on its own bootstrap sample, then averages independently-trained models. Dropout does not train separate networks. One shared set of weights gets exposed to an enormous number of different thinned subnetworks (for `n` neurons, `2^n` possible masks), and gradient updates from all of them accumulate into that same shared weight set. At inference, the model approximates averaging every one of those thinned subnetworks by running the full, undropped network once. Same ensembling spirit as Random Forest, different mechanism: weight sharing across masks instead of independently trained models.

## Train vs Inference: the Scaling Detail Keras Actually Uses

Dropout must behave differently at train time and test time, and getting the scaling direction right matters.

Keras implements **inverted dropout**, which moves the scaling to training time. During training, every activation that survives the mask gets divided by `(1-p)` (scaled up) to compensate for the fraction that got zeroed out, keeping the expected output magnitude constant. Since that correction already happened during training, inference needs no scaling at all: `Dropout` layers do nothing when the model runs in inference mode, and every neuron passes through unchanged. The two approaches land at the same expected output, just at opposite ends of training versus inference.

## Experiment 1: Regression

Same architecture, same data, only difference is two `Dropout(0.2)` layers:

```python
model_1 = Sequential()
model_1.add(Input(shape = (1,)))
model_1.add(Dense(128, activation="relu"))
model_1.add(Dense(128, activation="relu"))
model_1.add(Dense(1, activation="linear"))
model_1.compile(loss='mse', optimizer='Adam', metrics=['mse'])
history = model_1.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=False)
```

```python
model_2 = Sequential()
model_2.add(Input(shape = (1,)))
model_2.add(Dense(128, activation="relu"))
model_2.add(Dropout(0.2))
model_2.add(Dense(128, activation="relu"))
model_2.add(Dropout(0.2))
model_2.add(Dense(1, activation="linear"))
model_2.compile(loss='mse', optimizer='Adam', metrics=['mse'])
drop_out_history = model_2.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), verbose=False)
```

| | Train MSE | Test MSE | Gap (test - train) |
|---|---|---|---|
| No dropout | 0.007364 | 0.039318 | 0.031954 |
| Dropout 0.2 | 0.014336 | 0.034716 | 0.020380 |

Dropout makes training harder on purpose. Train MSE nearly doubles, from 0.0074 to 0.0143, because the network can't fit the 20 training points as tightly with a fifth of its neurons missing on every batch. Test MSE still improves, from 0.0393 to 0.0347, and the gap between train and test error shrinks by about a third (0.0320 to 0.0204). Worse training fit, better generalization: that trade is the entire point of dropout as a regularizer.

## Experiment 2: Classification

Same setup on a two-class 2D dataset, comparing decision boundaries and training curves with and without dropout:

```python
model = Sequential()
model.add(Input(shape = (2,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, validation_split=0.2, verbose=0)
```

```python
model = Sequential()
model.add(Input(shape = (2,)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))    # 40% of neurons dropped from the 1st hidden layer
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))    # 40% of neurons dropped from the 2nd hidden layer
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, validation_split=0.2, verbose=False)
```

This part of the notebook plots decision boundaries and loss/accuracy curves rather than printing final metrics, so there are no train/test numbers to report here the way Experiment 1 has. The no-dropout boundary traces the training points closely, including their noise; the 0.4-dropout boundary comes out smoother. That matches the mechanism above: a smoother boundary means the network leaned less on specific neuron combinations tied to individual points.