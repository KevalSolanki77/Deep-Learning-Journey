# Day 23 - Transfer Learning

## What It Is

Transfer learning (TL) takes knowledge gained on one problem (the source domain and task) and applies it to a different but related problem (the target domain and task). Instead of initializing a deep network with random weights and training from scratch, you start from the learned weights of a model pre-trained on a massive dataset such as ImageNet (images) or WebText (language).

In the standard formalism, a **domain** is a feature space plus its data distribution, $\mathcal{D} = \{\mathcal{X}, P(X)\}$, and a **task** is a label space plus a predictive function, $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$. Transfer learning improves the target function $f_T$ using knowledge from $(\mathcal{D}_S, \mathcal{T}_S)$, where the source and target differ in domain, in task, or in both.

In practice, this means one change to training. Plain training starts at random weights $\theta_0$. Transfer learning starts at pretrained weights $\theta_{pre}$, then runs the usual update:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}_{target}(\theta) \qquad \text{with } \theta \text{ initialized at } \theta_{pre}$$

The starting point does most of the work. A good $\theta_{pre}$ already sits in a region of weight space where useful image features live.

## Why It Helps

Deep networks learn representations in a hierarchy (Day 22 showed it on VGG16's feature maps):

| Layer depth | What it captures | Transfers to other tasks? |
|---|---|---|
| Early | Edges, color gradients, basic shapes (or basic syntax in language) | Yes, almost always |
| Middle | Motifs and textures built from those primitives | Often |
| Late | Task-specific concepts: leaf lesions, breeds, document classes | Seldom, so these layers get replaced |

Low and mid-level features are universal across visual and language tasks, which gives three advantages:

- **Less training time and compute.** The backbone is already optimized, so training converges in fewer epochs.
- **Works on small datasets.** A small labeled set cannot train millions of parameters without overfitting. A pretrained backbone shrinks the number of weights that need to learn from the target data.
- **Better generalization.** Features learned from millions of diverse samples act as an implicit regularizer (compare Days 9, 10, and 21).

## Core Strategies

| Strategy | Mechanism | Workflow | Primary use case |
|---|---|---|---|
| Feature extraction | The pretrained backbone is frozen (`trainable = False`) and acts as a fixed feature extractor | Strip the source classification head, append a new task-specific head (`GlobalAveragePooling2D` + `Dense`), and train only the head with an optimizer like Adam at $\text{lr} \approx 10^{-3}$ | Small target datasets, or target data very similar to the source |
| Fine-tuning | Part or all of the backbone is unfrozen and trained alongside the new head | Two stages: train the head first (warmup), then unfreeze the top $N$ layers and train with a tiny learning rate, $\text{lr} \approx 10^{-5}$, to prevent catastrophic forgetting | Moderate to large target datasets, or domains with subtle stylistic shifts |

### Why the Head Trains First

The new head starts with random weights, so its first gradients are large and meaningless. If the backbone were unfrozen at the same time, those gradients would flow into the pretrained layers and overwrite useful features with noise. That damage is **catastrophic forgetting**. Training the head first lets it settle, so the gradients that reach the backbone in stage two are small and informative.

### Why the Learning Rate Drops to 1e-5

Under Adam, each weight moves by at most about $\eta$ per step (Day 15). The notebook runs 125 steps per epoch for 5 epochs in each phase, so 625 steps:

| Phase | Learning rate | Upper bound on movement per weight over 625 steps |
|---|---|---|
| Feature extraction (head only) | $10^{-3}$ | $625 \times 10^{-3} = 0.625$ |
| Fine-tuning (block 5 + head) | $10^{-5}$ | $625 \times 10^{-5} = 0.00625$ |

The fine-tuning phase cannot move the pretrained weights far from $\theta_{pre}$, so it adjusts them without erasing them. This is a bound from the update rule, not a measurement of how far the weights moved.

## Decision Matrix: Dataset Size vs. Domain Similarity

The right configuration depends on the size of the target dataset and its similarity to the source data.

| | High similarity to source | Low similarity to source |
|---|---|---|
| **Small dataset** | **Feature extraction.** Freeze the entire backbone and train only the custom head. Reason: too little data to fine-tune without heavy overfitting. | **Challenging scenario.** Freeze early layers and train higher-level layers, or extract intermediate features. Heavy augmentation and regularization required. Reason: pretrained high-level features do not match, and data is scarce. |
| **Large dataset** | **Fine-tuning (top blocks or full network).** Fine-tune the top layers with a small learning rate, or initialize the full network with pretrained weights. Reason: abundant data enables rapid convergence without overfitting. | **Full-model fine-tuning.** Retrain most or all layers, using the pretrained weights as a robust initialization instead of random noise. Reason: ample data allows adaptation of both generic and specific features. |

## Common Pre-Trained Models

**Computer vision**

| Model | Notes |
|---|---|
| MobileNet (V2 / V3) | Lightweight; depthwise-separable convolutions built for mobile and edge inference and fast training |
| ResNet (50 / 101) | Residual (skip) connections mitigate vanishing gradients; robust general-purpose baseline |
| EfficientNet (B0 to B7, V2) | Compound scaling across depth, width, and resolution gives state-of-the-art parameter efficiency |
| Vision Transformers (ViT, Swin) | Self-attention across image patches; excel on very large benchmark distributions |

**Natural language processing**

| Model | Notes |
|---|---|
| BERT / RoBERTa / DeBERTa | Masked language models with bidirectional contextual features: classification, NER, extractive QA |
| T5 / BART | Encoder-decoder models for translation, summarization, and text transformation |
| LLaMA / Mistral / GPT family | Autoregressive decoder-only foundation models for generation and reasoning |

**Audio and speech**

| Model | Notes |
|---|---|
| Wav2Vec 2.0 / Whisper | Self-supervised and weakly supervised multilingual speech recognition and audio-feature backbones |

## The Experiment: Cats vs Dogs with VGG16

The notebook applies both strategies in order on a binary image task: feature extraction for 5 epochs, then fine-tuning the last convolutional block for 5 epochs.

Where it sits in the matrix: ImageNet already contains many dog breeds and several cat classes, so this target is about as similar to the source as a target can be, and 8,000 training images sit between "small" and "large." Keep that in mind when reading the results. They show the method works on a close domain and say little about distant ones.

### Setup: Data

The dataset is `chetankv/dogs-cats-images` from Kaggle (CC0 license), 435 MB.

```python
import os
import tensorflow as tf
import keras
from keras import layers, models

DATASET_PATH = "chetankv/dogs-cats-images"
```

```python
!kaggle datasets download -d {DATASET_PATH}
```

```python
import zipfile
zip_file_name = DATASET_PATH.split('/')[-1] + ".zip"

if os.path.exists(zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall("/content/dataset")
    print("Dataset successfully downloaded and unzipped into /content/dataset")
    # Clean up the zip file to save space
    os.remove(zip_file_name)
else:
    print("Download failed. Please check your Kaggle path.")
```

```python
DATA_DIR = "/content/dataset/dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
```

`IMG_SIZE = (224, 224)` matches the size VGG16 was trained on.

```python
train_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/training_set",
    seed=1,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test_set",
    seed=1,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
```

```text
Found 8000 files belonging to 2 classes.
Found 2000 files belonging to 2 classes.
```

Labels come from the folder names in alphabetical order, so `cats` is class 0 and `dogs` is class 1, and the sigmoid output is the probability of "dog." With 8,000 images and a batch size of 64, one epoch is 125 steps. The 2,000 validation images make each single error worth 0.05 accuracy points.

```python
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

`cache()` keeps the decoded batches in memory after the first epoch, and `prefetch()` prepares the next batch while the GPU works on the current one. The timing in the results confirms it: epoch 1 took 64 s, and later epochs took 38 to 42 s.

### Setup: The Frozen Backbone

```python
from keras.applications.vgg16 import VGG16, preprocess_input

base_model = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
```

Two arguments define the strategy:

- `include_top=False` strips the original 1,000-class classifier, including the three dense layers that held 89% of VGG16's parameters (Day 22). What remains is the 13-layer convolutional base with **14,714,688 parameters** and an output of 7×7×512.
- `base_model.trainable = False` freezes every one of those weights. The summary shows 14,714,688 non-trainable and 0 trainable.

### Setup: The Full Model

```python
model = models.Sequential([
    layers.Input(shape = (224, 224, 3)),
    layers.Lambda(preprocess_input), # Scale : 0-255 -> 0-1
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
```

| Layer | Output shape | Parameters | Arithmetic |
|---|---|---|---|
| Lambda (`preprocess_input`) | 224×224×3 | 0 | |
| VGG16 base (frozen) | 7×7×512 | 14,714,688 | |
| GlobalAveragePooling2D | 512 | 0 | |
| Dense 128, ReLU | 128 | 65,664 | 512·128 + 128 |
| Dropout 0.2 | 128 | 0 | |
| Dense 1, sigmoid | 1 | 129 | 128 + 1 |

Total **14,780,481** parameters: **65,793 trainable** (0.45%) and 14,714,688 non-trainable.

The comment on the `Lambda` line says the layer scales pixels from 0-255 to 0-1. It does not. For VGG16, `preprocess_input` converts RGB to BGR and subtracts the ImageNet channel means (103.939, 116.779, 123.68) with no division by 255, as covered on Day 22. Pretrained weights expect that exact preprocessing, and the `Lambda` layer applies it inside the model so inference code cannot skip it.

The head is the Day 19 pattern: global average pooling replaces `Flatten` + a huge dense layer, which keeps the head at 65,793 parameters. A `Flatten` on 7×7×512 would feed 25,088 values into `Dense(128)` and cost 3.2 million parameters.

## Phase 1: Feature Extraction (Train Only the Head)

```python
model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    batch_size=BATCH_SIZE
)
```

| Epoch | Train acc | Train loss | Val acc | Val loss | Time |
|---|---|---|---|---|---|
| 1 | 0.9467 | 0.1924 | 0.9800 | 0.0759 | 64 s |
| 2 | 0.9794 | 0.0599 | 0.9785 | 0.0682 | 38 s |
| 3 | 0.9881 | 0.0319 | 0.9785 | 0.0770 | 39 s |
| 4 | 0.9916 | 0.0231 | 0.9770 | 0.0802 | 41 s |
| 5 | 0.9940 | 0.0180 | 0.9770 | 0.0793 | 42 s |

**One epoch on 65,793 trainable weights reached 98.0% validation accuracy.** The frozen VGG16 features already separate cats from dogs, and the head only has to read them out. After epoch 1 the head overfits. Training accuracy climbs from 0.9467 to 0.9940 while validation accuracy slips from 0.9800 to 0.9770, and validation loss bottoms out at epoch 2 (0.0682) and rises to 0.0793. Each epoch makes the head fit the training images better and the validation images a little worse.

## Phase 2: Fine-Tuning (Unfreeze the Last Block)

Unfreeze the last convolutional block (`block5_conv1`, `block5_conv2`, `block5_conv3`, and `block5_pool`) so the model can adjust high-level features for this dataset, and drop the learning rate to $10^{-5}$.

```python
# Unfreeze the base model
base_model.trainable = True

# Freeze all layers except the last block (block5)
# VGG16 has 19 layers (including input/pooling). We can freeze up to 'block5_conv1'
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

# Verify trainable status
for i, layer in enumerate(base_model.layers):
    print(f"{i}: {layer.name} - Trainable: {layer.trainable}")

# Compile the model with a very low learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Summary to show the updated trainable parameters count
model.summary()
```

The check loop prints `False` for layers 0 to 14 (input through `block4_pool`) and `True` for layers 15 to 18 (`block5_conv1` through `block5_pool`).

| | Phase 1 | Phase 2 |
|---|---|---|
| Trainable | 65,793 (0.45%) | 7,145,217 (48.3%) |
| Non-trainable | 14,714,688 | 7,635,264 |
| Learning rate | 1e-3 (Adam default) | 1e-5 |

The three block-5 convolutions hold $3 \times 2{,}359{,}808 = 7{,}079{,}424$ parameters, which is 48.1% of the backbone. Add the head's 65,793 to reach 7,145,217 trainable.

Two details in that cell matter:

- **The `compile` call is required.** A change to `trainable` takes effect only when the model is compiled again. Without it, training would continue with the Phase 1 freeze pattern.
- **The new `Adam` starts fresh.** Compiling creates a new optimizer with its first and second moment buffers at zero (Day 15), so the Phase 1 momentum history does not carry over. Adam's bias correction handles that cold start.

```python
# Fine-tune the model for 5 epochs
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)
```

| Epoch | Train acc | Train loss | Val acc | Val loss | Time |
|---|---|---|---|---|---|
| 1 | 0.9955 | 0.0138 | 0.9800 | 0.0682 | 51 s |
| 2 | 0.9964 | 0.0115 | 0.9850 | 0.0685 | 45 s |
| 3 | 0.9991 | 0.0036 | 0.9860 | 0.0694 | 46 s |
| 4 | 0.9989 | 0.0032 | 0.9875 | 0.0614 | 47 s |
| 5 | 0.9999 | 0.0013 | 0.9880 | 0.0660 | 49 s |

Epoch numbering restarts at 1 because the second `fit` call has no `initial_epoch`. The model weights carry over from Phase 1.

### Plot Both Phases on One Axis

```python
import matplotlib.pyplot as plt

# Retrieve metrics
acc_phase1 = history_phase1.history['accuracy']
val_acc_phase1 = history_phase1.history['val_accuracy']

acc_phase2 = history_phase2.history['accuracy']
val_acc_phase2 = history_phase2.history['val_accuracy']

# Combine metrics
total_acc = acc_phase1 + acc_phase2
total_val_acc = val_acc_phase1 + val_acc_phase2

# Plot configuration
plt.figure(figsize=(10, 6))
plt.plot(total_acc, label='Training Accuracy', color='blue', marker='o')
plt.plot(total_val_acc, label='Validation (Testing) Accuracy', color='orange', marker='s')

# Draw a vertical line to mark the transition to fine-tuning
plt.axvline(x=len(acc_phase1) - 0.5, color='red', linestyle='--', label='Start of Fine-Tuning')

plt.title('Training vs Testing Accuracy (Feature Extraction & Fine-Tuning)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

The plot concatenates the two history lists into 10 points, with a dashed red line between epochs 5 and 6 marking where fine-tuning starts.

## Reading the Results

| Checkpoint | Val acc | Errors out of 2,000 |
|---|---|---|
| Phase 1, epoch 1 (best of phase 1) | 0.9800 | 40 |
| Phase 1, epoch 5 (end) | 0.9770 | 46 |
| Phase 2, epoch 1 | 0.9800 | 40 |
| Phase 2, epoch 5 (end) | 0.9880 | 24 |

**Fine-tuning improved validation accuracy by 1.1 points over the end of Phase 1.** Validation accuracy rose in every Phase 2 epoch (0.9800, 0.9850, 0.9860, 0.9875, 0.9880), and the model made 24 errors against 46 at the end of Phase 1. Against the best Phase 1 epoch (40 errors), the gain is 16 errors, or 0.8 points. With 2,000 validation images, each accuracy estimate carries a sampling error of about ±0.5 points (±0.48 at 98.8%, ±0.66 at 97.7%), so the gain is larger than that noise but the evidence is one run, not a verdict.

**Fine-tuning improved accuracy more than it improved loss.** Validation loss moved from 0.0682 to 0.0660, with a best of 0.0614 at epoch 4, which is flat compared to the accuracy change. Meanwhile training loss fell from 0.0138 to 0.0013. The model fits the training set ever more tightly while its confidence on validation images does not improve. The training-validation accuracy gap is 1.2 points at the end of Phase 2 (0.9999 against 0.9880), down from 1.7 points at the end of Phase 1 (0.9940 against 0.9770). Training accuracy is the running average during each epoch with Dropout active, so it understates the model's clean-image training accuracy slightly.

**Fine-tuning cost about 19% more time per epoch.** Phase 2 epochs averaged 47.6 s (45 to 51 s) against 40.0 s (38 to 42 s) for Phase 1 epochs after the cache filled. Every epoch still needs a forward pass through the whole backbone, and Phase 2 adds backpropagation through block 5.

**The notebook has no from-scratch baseline,** so it does not measure how much transfer learning helped relative to training a CNN from random weights. The evidence for the benefit in this notebook is the speed: 98.0% after one epoch and 64 s of training.

## Notes on the Notebook

- **The validation set is the test set.** `val_ds` loads `test_set`, and the notebook uses it both to watch training and to report the final number. No third, untouched split exists. This notebook tried one configuration, so the bias is small, but a held-out test set is the standard way to report a final result.
- **`.cache()` after `image_dataset_from_directory` freezes the batch order.** The shuffle happens once, in the first epoch. Every later epoch replays the same cached batches in the same order. Caching also stores the images as float32, which comes to about 4.5 GiB for the training set and 1.1 GiB for the validation set (8,000 × 224 × 224 × 3 × 4 bytes). That fits a 12 GB Colab session and can run out of memory on a smaller machine.
- **The `# Scale : 0-255 -> 0-1` comment on `Lambda` is wrong.** The layer applies VGG16's own preprocessing (BGR conversion and mean subtraction).
- **`batch_size=BATCH_SIZE` in the first `fit` call does nothing.** The dataset is already batched at 64.
- **VGG16 has no BatchNormalization layers,** so unfreezing block 5 is safe as written. With ResNet or EfficientNet, BatchNorm layers need care during fine-tuning: keep them in inference mode so their running statistics do not shift on a small batch (Day 13).

## Real Use

| Domain | How transfer learning applies |
|---|---|
| Crop disease detection | Leaf-image classifiers start from an ImageNet backbone because labeled disease photos are scarce and expensive to collect |
| Medical imaging | Chest X-ray and skin-lesion classifiers fine-tune ImageNet backbones, since specialist labeling limits dataset size |
| Defect detection | Factory-line inspection models start from a pretrained backbone and learn the few defect classes from hundreds of images |
| Indian-language NLP | AI4Bharat's IndicBERT, pretrained on Indian languages, is fine-tuned for classification in those languages |
| Speech recognition | Whisper checkpoints fine-tuned on Hindi and other Indian-language speech are shared on Hugging Face |

## One-Line Takeaways

- Transfer learning starts training from pretrained weights instead of random ones, and the starting point does most of the work.
- Early layers transfer almost always, and late layers are the ones to replace.
- Feature extraction trains only the new head. Fine-tuning then unfreezes the top block with a tiny learning rate.
- Train the head first, or random head gradients damage the backbone.
- Recompile after changing `trainable`, or the change does not apply.
- In this notebook, 65,793 trainable weights reached 98.0% validation accuracy in one epoch, and fine-tuning block 5 lifted it to 98.8%.
- Cats and dogs sit close to ImageNet, so these results show the method on an easy domain. A distant domain falls in the right-hand column of the decision matrix, where the recipe changes.