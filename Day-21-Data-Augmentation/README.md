# Day 21 (Deep Learning) - Image Data Augmentation

## What It Is

Data augmentation applies random, label-preserving transformations to training images, so the model sees a different version of each image every epoch. It works as a regularizer. A model that sees the same 5,000 images forever can memorize pixel positions. A model that sees each image shifted, tilted, and rescaled has to learn what defines the class.

**The one rule: the transformation must not change the label.**

Training on augmented data minimizes the loss averaged over a distribution $T$ of transformations:

$$\min_\theta \; \mathbb{E}_{(x,y)} \; \mathbb{E}_{t \sim T} \Big[ \mathcal{L}\big(f_\theta(t(x)),\, y\big) \Big] \qquad \text{valid only if } y(t(x)) = y(x)$$

Each transformation $t$ encodes one piece of prior knowledge: "a cat is still a cat when the photo is tilted." The network would need thousands of extra labeled photos to learn that fact from data. Augmentation hands it over for free.

### Why It Regularizes

Day 9 (Dropout) and Day 10 (L1/L2) attacked overfitting by adding noise to activations and penalizing large weights. Augmentation attacks the same variance problem from the input side. For small Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ added to the input, a second-order expansion with squared-error loss gives (Bishop, 1995):

$$\mathbb{E}_\epsilon\big[(f(x+\epsilon) - y)^2\big] \approx (f(x)-y)^2 + \sigma^2 \,\lVert \nabla_x f(x) \rVert^2 + (f(x)-y)\,\sigma^2\, \mathrm{tr}\,\nabla_x^2 f(x)$$

The middle term penalizes a function that changes fast when the input moves a little. The last term averages out when $f(x)$ approximates the true conditional mean. Training with input noise therefore acts like a smoothness penalty on the model, the same job L2 does for weights. Geometric transformations extend the idea: the penalty applies along the directions of rotation, shift, and scale, not just along random noise.

## The Main Families

| Type | Examples | Invariance it teaches |
|---|---|---|
| Geometric | flip, rotate, crop, zoom, translate | position, orientation, scale |
| Photometric | brightness, contrast, saturation, hue | lighting, camera differences |
| Noise / blur | Gaussian noise, blur, JPEG artifacts | sensor quality |
| Occlusion | Cutout, Random Erasing | partially hidden objects |
| Mixing | Mixup, CutMix | smoother decision boundaries |
| Learned policies | AutoAugment, RandAugment | automatic selection of transforms |

### The Math Behind Each Family

**Flip (horizontal).** Mirror the columns of an image of width $W$:

$$x'_{i,j} = x_{i,\,W-1-j}$$

**Rotation.** Rotate each pixel coordinate $p$ around the image center $c$ by angle $\theta$:

$$p' = R(\theta)(p - c) + c \qquad R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Zoom.** Scale coordinates around the center by a factor $s$: $p' = c + s\,(p - c)$. Zooming out leaves empty borders that need filling, which is what `fill_mode` controls (see the bug notes below).

**Contrast.** Stretch pixel values around the image mean $\bar{x}$:

$$x' = (x - \bar{x})\,\alpha + \bar{x}$$

**Brightness and noise.** Shift or perturb every pixel: $x' = x + \delta$ for brightness, $x' = x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ for sensor noise.

**Cutout / Random Erasing.** Zero out (or fill with noise) a random rectangle. The label stays the same, so the model must recognize the object from its remaining parts.

**Mixup.** Blend two images and blend their labels by the same weight:

$$\tilde{x} = \lambda x_i + (1-\lambda)\,x_j \qquad \tilde{y} = \lambda y_i + (1-\lambda)\,y_j \qquad \lambda \sim \mathrm{Beta}(\alpha,\alpha)$$

**CutMix.** Paste a rectangle from image $j$ into image $i$ and mix the labels by the pasted area. With binary mask $M$ and $\lambda$ equal to the fraction of pixels kept from $x_i$:

$$\tilde{x} = M \odot x_i + (1-M) \odot x_j \qquad \tilde{y} = \lambda y_i + (1-\lambda)\,y_j$$

Mixup and CutMix break the "one rule" on purpose: the label changes, but it changes in a controlled way, and the soft label tells the model how uncertain it should be.

**Learned policies.** AutoAugment searches over sequences of transforms with reinforcement learning. RandAugment drops the search and uses two knobs: how many transforms to apply per image and how strong each one is.

## In Keras

The summary's pipeline:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])
```

Put it inside the model. These layers run only during training and switch off at inference, so `model.evaluate()` and `model.predict()` see the original images. Keras handles the switch through the `training` argument, the same mechanism that controls Dropout (Day 9) and BatchNorm (Day 13).

**What the factors mean.** Keras measures rotation as a fraction of a full turn, so the angle is drawn from a uniform range:

$$\theta \sim U(-2\pi f,\; 2\pi f)$$

| `RandomRotation(f)` | Angle range |
|---|---|
| 0.05 | ±18° |
| 0.1 | ±36° |
| 0.2 | ±72° |

`RandomZoom(0.2)` scales the content by a random factor of about 0.8x to 1.2x. `RandomContrast(0.2)` draws $\alpha$ from $[0.8, 1.2]$.

Recent Keras 3 releases also ship `RandAugment`, `RandomErasing`, `MixUp`, and `CutMix` layers. Check the installed version before relying on them.

### Online vs Offline

| | Online | Offline |
|---|---|---|
| How | Random transform applied on the fly each epoch | Augmented copies saved to disk before training |
| Variety | Different every epoch, unbounded | Fixed set of copies |
| Storage | None | Multiplies the dataset size |
| Use when | Default for almost everything | Expensive transforms, or you need to inspect the output |

The notebook uses the offline route for its first experiment, because the goal is to look at the output.

## The Experiment

### Part 1: Augment One Image and Look at It

Load a cat photo at 200×200 and turn it into a batch of one image:

```python
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras import layers, models

img = image.load_img(
    path = "data/cat.jpg",
    target_size=(200, 200)
)
```

Keras layers expect a batch dimension, so the array gets shape `(1, 200, 200, 3)`. The pipeline runs 10 times, and each run draws new random parameters. Each result goes to disk:

```python
# Prepare the image array
img_array = image.img_to_array(img)

# Add a batch dimension -> (1, height, width, channels)
img_array = tf.expand_dims(img_array, axis=0)
data_augmentation = models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

for i in range(10):
    # Apply augmentation (calling the pipeline directly handles the batch)
    augmented_batch = data_augmentation(img_array)

    # Remove the batch dimension to get back a 3D array -> (height, width, channels)
    # Also ensure the values are converted to integers/uint8 format for saving
    new_img_array = tf.squeeze(augmented_batch, axis=0).numpy().astype("uint8")

    # Save the array directly as an image file
    image.save_img(f"augmented_data/aug_img_{i}.jpg", new_img_array)
```

Calling the pipeline outside `fit` works because these layers default to `training=True` when called outside `fit`. The `augmented_data/` folder must exist before the loop runs, or the save fails.

Display the 10 saved images in a grid:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Flatten the 2D array of axes to a 1D list so it's easier to loop through
axes = axes.flatten()

# 3. Loop through and plot each of the 10 images
for i in range(10):
    # Construct the exact file path matching the saved names
    img_path = f"augmented_data/aug_img_{i}.jpg"

    # Load the image using Keras utility
    img = image.load_img(img_path)

    # Display the image on the corresponding subplot axis
    axes[i].imshow(img)
    axes[i].set_title(f"Augmented {i+1}")
    axes[i].axis('off')  # Turn off the pixel coordinate axes for a cleaner look

# 4. Automatically adjust layout spacing and show the plot
plt.tight_layout()
plt.show()
```

### What the 10 Samples Show

- **The variety is large.** Samples 1, 4, 5, 6, and 7 stay close to the original pose. Samples 2, 3, 8, 9, and 10 show strong tilt and zoom. Sample 9 has the cat rotated by about 70°, near the ±72° ceiling that `RandomRotation(0.2)` allows.
- **The label survives.** Every sample is still a cat, so the transformations meet the one rule.
- **The strongest samples sit at the edge of realism.** A phone photo of a cat tilted 70° is rare. Rotation at 0.2 spends training signal on poses the model will almost never meet at deployment. The summary's 0.1 (±36°) is the safer starting point, and 0.05 suits datasets where the camera stays level.
- **Border artifacts appear.** Samples 3, 7, and 10 show a small orange fragment in the top-left corner. Rotation and zoom-out leave empty regions, and the default `fill_mode="reflect"` fills them by mirroring the image content, which here pastes a ghost piece of the cat's ear into the corner. Options: `"constant"` (solid fill), `"nearest"` (smears the edge pixel), `"wrap"` (tiles the opposite edge), or `"reflect"` (mirror, the default). For a centered object on a plain background, `"constant"` or `"nearest"` avoids the ghost fragments.

### Part 2: Put Augmentation Inside the CNN

The model reuses Day 20's CNN, with `data_augmentation` as the first layer after the input and `Rescaling` right after it. Augmentation runs on raw 0-255 pixel values, then rescaling maps them to [0, 1].

```python
CNN = keras.Sequential([

    layers.Input(shape = (200, 200, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, kernel_size=(3,3),padding="valid",activation="relu",input_shape=(256,256,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2),
    layers.Dropout(0.3),

    layers.Conv2D(64, kernel_size=(3,3),padding="valid",activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2),
    layers.Dropout(0.3),

    layers.Conv2D(128, kernel_size=(3,3),padding="valid",activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64,activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1,activation="sigmoid")

])
```

The model stacks five ideas from earlier days: Conv + pooling (Day 19), BatchNorm (Day 13), Dropout (Day 9), a GAP head (Day 19), and now augmentation.

**Shape trace** with the formula $\lfloor (n-k)/s \rfloor + 1$:

| Layer | Output shape |
|---|---|
| Input / augmentation / rescaling | 200×200×3 |
| Conv 32, 3×3, valid | 198×198×32 |
| MaxPool 2×2 | 99×99×32 |
| Conv 64, 3×3, valid | 97×97×64 |
| MaxPool 2×2 | 48×48×64 |
| Conv 128, 3×3, valid | 46×46×128 |
| MaxPool 2×2 | 23×23×128 |
| GlobalAveragePooling2D | 128 |
| Dense 128, Dense 64, Dense 1 | 128, 64, 1 |

**Parameter count**, computed from the formulas because the notebook never calls `model.summary()`:

| Layer | Arithmetic | Parameters |
|---|---|---|
| Conv 32 | 3·3·3·32 + 32 | 896 |
| BatchNorm (32) | 4 × 32 | 128 |
| Conv 64 | 3·3·32·64 + 64 | 18,496 |
| BatchNorm (64) | 4 × 64 | 256 |
| Conv 128 | 3·3·64·128 + 128 | 73,856 |
| BatchNorm (128) | 4 × 128 | 512 |
| Dense 128 | 128·128 + 128 | 16,512 |
| BatchNorm (128) | 4 × 128 | 512 |
| Dense 64 | 128·64 + 64 | 8,256 |
| BatchNorm (64) | 4 × 64 | 256 |
| Dense 1 | 64 + 1 | 65 |

Total 119,745: 118,913 trainable and 832 non-trainable (BatchNorm running means and variances). Augmentation, rescaling, pooling, and dropout layers add no parameters.

### Two Leftovers in the Notebook

- **`input_shape=(256,256,3)` on the first `Conv2D` does nothing.** The model already begins with `Input(shape=(200, 200, 3))`, so the argument has no effect. It is a leftover from Day 20, where the input was 256×256. Delete it to avoid confusion.
- **The notebook stops before training.** The model is defined but never compiled or fit, so this README has no accuracy numbers to report and makes no claim about how much augmentation helps this model. The check below describes what to measure once a training run exists.

## Common Mistakes

| Mistake | Why it hurts | Example |
|---|---|---|
| Augmenting validation or test data | Metrics measure the augmented distribution instead of the real one, and runs stop being comparable | Keras avoids this when augmentation sits inside the model, because evaluation turns it off |
| Label-changing transforms | The model learns wrong labels | Flips or 180° rotations on digits (a rotated 6 looks like a 9), on text, and on medical scans where left and right matter |
| Over-augmenting | Training images that never occur in reality waste capacity and can lower accuracy | Rotation at ±72° on phone photos of cats |
| Augmenting before the split | Copies of one image land in both train and validation, which leaks information and inflates validation accuracy | Run `train_test_split` first, then augment only the training side |

The offline route carries the leakage risk most often. Online augmentation inside the model cannot leak, because the split already happened and the augmentation only touches what flows through the training loop.

## Real Use

| Domain | Why augmentation matters there |
|---|---|
| Medical imaging | Datasets are tiny because labeling needs a specialist. Small rotations, shifts, and contrast changes mimic scanner and patient-positioning variation. Left-right flips can corrupt laterality, so check the task first |
| Manufacturing defect detection | Defects are rare, so the defect class has few examples. Rotation, flips, and brightness changes multiply them. The NEU steel-surface defect dataset has only 300 images per defect class, which is the scale where augmentation matters |
| Autonomous driving | Synthetic rain, fog, and night shift the training distribution toward conditions that are rare in recorded data but dangerous on the road |

## Is It Working?

The notebook has no training run, so these are the diagnostics to apply once one exists.

| What the curves show | Meaning | Action |
|---|---|---|
| Training accuracy rises more slowly, sits closer to validation accuracy, and validation accuracy ends higher | Augmentation is doing its job | Keep it |
| Large train-validation gap | The model still memorizes | Add or strengthen augmentation (also see Days 9 and 10) |
| Both training and validation accuracy are low | Underfitting | Reduce augmentation strength or enlarge the model |

Training accuracy measured on augmented batches reads lower than it would on clean images, because the model is graded on harder inputs. A lower training curve is the expected signature, not a sign that training broke.

## One-Line Takeaways

- Augmentation shows the model a new version of each training image every epoch and acts as a regularizer, like Dropout and L2.
- The rule: the transformation must keep the label true.
- Keras random layers go inside the model and turn off at inference.
- `RandomRotation(f)` covers ±$360° \times f$, so 0.2 allows ±72°.
- The default `fill_mode="reflect"` can paste ghost fragments into the corners, as the sample grid shows.
- Augment only the training data, and split before any offline copying.