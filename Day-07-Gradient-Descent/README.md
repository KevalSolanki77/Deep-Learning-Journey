# Day 7 : Batch vs Stochastic vs Mini-Batch Gradient Descent

### Link : [Notebook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-07-Gradient-Descent/GD.ipynb)

---

## What Each Variant Does

All three update the same weights with the same gradient descent rule. They differ only in how much data goes into computing one gradient before an update happens.

- **Batch GD**: one update per epoch, computed from the full dataset.
- **Stochastic GD (SGD)**: one update per sample.
- **Mini-Batch GD**: one update per small batch, the middle ground between the two, and the default in practice.

## Setup

`fetch_california_housing`, 20,640 rows, 8 numeric features, split 80/20, scaled with `StandardScaler`. 16,512 rows land in the training set, the number behind every "updates per epoch" figure below.

```python
df = fetch_california_housing(as_frame=True)['frame']

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['MedHouseVal']), df['MedHouseVal'], test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Experiment 1: Training Time

```python
batch = Sequential()
batch.add(Input(shape = (X_train.shape[1],)))
batch.add(Dense(32, activation='relu'))
batch.add(Dense(16, activation='relu'))
batch.add(Dense(1))
batch.compile(loss='mse')

start = time.time()
batch.fit(X_train, y_train, epochs=10, batch_size=len(X_train)) # one update per epoch
end = time.time()
batch_time = end - start

stocastic = Sequential()
stocastic.add(Input(shape = (X_train.shape[1],)))
stocastic.add(Dense(32, activation='relu'))
stocastic.add(Dense(16, activation='relu'))
stocastic.add(Dense(1))
stocastic.compile(loss='mse')

start = time.time()
stocastic.fit(X_train, y_train, epochs=1, batch_size=1) # one update per sample
end = time.time()
stocastic_time = end - start
```

| | Epochs | Updates per epoch | Total time | Final loss |
|---|---|---|---|---|
| Batch GD | 10 | 1 | 1.04s | 1.9443 |
| Stochastic GD | 1 | 16,512 | 27.77s | 0.8826 |

Batch GD finishes 10 full passes over the data faster than SGD finishes one. The reason isn't the amount of arithmetic. It's how that arithmetic runs: batch GD computes one gradient from a single matrix multiply across all 16,512 rows, while SGD forces 16,512 separate Python-level steps in that one epoch. Vectorized dot products beat a loop over individual samples on the same hardware, every time.

## Experiment 2: Convergence at Equal Epoch Count

```python
batch = Sequential()
batch.add(Input(shape = (X_train.shape[1],)))
batch.add(Dense(32, activation='relu'))
batch.add(Dense(16, activation='relu'))
batch.add(Dense(1))
batch.compile(loss='mse')

start = time.time()
batch_history = batch.fit(X_train, y_train, epochs=3, batch_size=len(X_train))
end = time.time()
batch_time = end - start

stocastic = Sequential()
stocastic.add(Input(shape = (X_train.shape[1],)))
stocastic.add(Dense(32, activation='relu'))
stocastic.add(Dense(16, activation='relu'))
stocastic.add(Dense(1))
stocastic.compile(loss='mse')

start = time.time()
stocastic_history = stocastic.fit(X_train, y_train, epochs=3, batch_size=1)
end = time.time()
stocastic_time = end - start
```

| | Epochs | Updates total | Total time | Final loss |
|---|---|---|---|---|
| Batch GD | 3 | 3 | 1.12s | 3.4427 |
| Stochastic GD | 3 | 49,536 | 103.15s | 0.3549 |

Matching the epoch count here doesn't match the update count. Batch GD ran 3 gradient updates total. SGD ran 49,536. Lower loss for SGD says more about running 16,512 times as many weight updates than it says about stochastic updates converging faster step-for-step.

## The Middle Ground: Mini-Batch GD

```python
mini_batch = Sequential()
mini_batch.add(Input(shape = (X_train.shape[1],)))
mini_batch.add(Dense(32, activation='relu'))
mini_batch.add(Dense(16, activation='relu'))
mini_batch.add(Dense(1))
mini_batch.compile(loss='mse')

start = time.time()
batch_history = mini_batch.fit(X_train, y_train, epochs=10, batch_size=512)
end = time.time()
mini_batch_time = end - start
```

| | Epochs | Updates per epoch | Total time | Final loss |
|---|---|---|---|---|
| Mini-Batch GD (512) | 10 | 33 | 2.68s | 0.4245 |

33 updates per epoch (16,512 rows / 512 per batch, rounded up for the final partial batch) instead of 1. Each update still runs as a vectorized matrix operation, just over 512 rows instead of 16,512 or 1. Ten epochs finish in 2.68 seconds, close to Batch GD's per-epoch cost, and the loss after 10 epochs (0.4245) lands well below either single-pass extreme. More frequent updates than full-batch, each one still cheap enough to vectorize: that combination is why mini-batch is the default in practice, not a compromise picked for lack of a better option.

## Key Takeaways

- **Vectorization** is why batch size changes speed at all. A dot product across many rows runs as one hardware-level operation; a Python loop over individual samples can't use that path, which is the entire gap between the 1.04s and 27.77s runs above.
- **Batch sizes as powers of 2** (32, 64, 128, 512) match how memory allocates on GPU/CPU hardware, which is why `batch_size=512` shows up here rather than an arbitrary number like 500.
- **Uneven division** between dataset size and batch size just leaves a smaller final batch. 16,512 / 512 = 32.25, so the mini-batch run above processes 32 full batches of 512 and one partial batch of 128, no special handling required.

### Link : [Notebook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-07-Gradient-Descent/GD.ipynb)