# Day 22 - Pretrained Models & Visualizing What a CNN Sees

## Why Use a Pretrained Model

Two problems push you toward a pretrained model.

**1. Deep learning is data hungry.** A network with millions of parameters needs a very large labeled dataset, and most projects do not have one. VGG16 has 138,357,544 parameters. Train that from scratch on 2,000 photos and it memorizes them. VGG16 learned from ImageNet: 1.28 million training images across 1,000 classes, labeled by people at a scale no student project can match.

**2. Training time.** Designing your own network and training it takes hours, days, or weeks. The VGG authors reported 2 to 3 weeks of training on 4 GPUs for their models. A pretrained model hands you the result of that run for free.

### Why It Works: Features Transfer

The second half of this entry gives the evidence. The first convolution layers of VGG16 learn edge and intensity detectors. Every photo, whether of cats, tumors, or scratches on steel, contains edges, so those layers apply to any image task. Only the last layers specialize to the original 1,000 classes. Reuse the general layers, replace the specialized ones, and a small dataset goes much further.

### How Pretrained Models Get Used

The notebook loads VGG16 to inspect it and does not retrain it. The table covers the three usual ways to apply one.

| Strategy | What you do | Use when |
|---|---|---|
| Feature extraction | Freeze the pretrained convolutional base, train a new classifier head on top | Small dataset, similar to ImageNet |
| Fine-tuning | Freeze the early blocks, unfreeze the last blocks, train with a small learning rate | Medium dataset, or a domain that differs from ImageNet |
| Train from scratch | Ignore the weights | Large dataset in a very different domain |

Real cases: chest X-ray and skin-lesion classifiers often start from ImageNet-pretrained backbones because labeled medical images are scarce. Plant-disease classifiers, defect detectors on factory lines, and product-image tagging follow the same pattern.

### Common Backbones in Keras

| Model | Parameters (with classifier) |
|---|---|
| VGG16 | 138.4 M |
| ResNet50 | about 25.6 M |
| EfficientNetB0 | about 5.3 M |
| MobileNetV2 | about 3.5 M |

VGG16 is the oldest and largest of the four, and its plain stack of 3×3 convolutions makes it the easiest to read. That is why it suits a visualization exercise.

## VGG16

```python
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16, preprocess_input

model = VGG16()
model.summary()
```

`VGG16()` with no arguments downloads the ImageNet weights and keeps the full classifier: input 224×224×3, 1,000 softmax outputs.

### Architecture

| Block | Layers | Output shape | Parameters |
|---|---|---|---|
| 1 | conv 64, conv 64, pool | 224×224×64, then 112×112×64 | 1,792 + 36,928 |
| 2 | conv 128, conv 128, pool | 112×112×128, then 56×56×128 | 73,856 + 147,584 |
| 3 | conv 256 ×3, pool | 56×56×256, then 28×28×256 | 295,168 + 590,080 + 590,080 |
| 4 | conv 512 ×3, pool | 28×28×512, then 14×14×512 | 1,180,160 + 2,359,808 + 2,359,808 |
| 5 | conv 512 ×3, pool | 14×14×512, then 7×7×512 | 2,359,808 × 3 |
| Head | Flatten (25,088), Dense 4096, Dense 4096, Dense 1000 | 1000 | 102,764,544 + 16,781,312 + 4,097,000 |

Every convolution is 3×3 with `same` padding and ReLU, so only the pools shrink the map. Each conv count follows $k_h k_w C_{in} C_{out} + C_{out}$. For `block1_conv1`: $3 \cdot 3 \cdot 3 \cdot 64 + 64 = 1{,}792$.

The summary reports **138,357,544 parameters (527.79 MB)**, all trainable.

**The dense head holds 89% of the model.** The 13 convolution layers sum to 14,714,688 parameters (10.6%). The three dense layers hold 123,642,856 (89.4%), and `fc1` alone holds 102,764,544 because it connects 7×7×512 = 25,088 flattened values to 4,096 units. This repeats the LeNet-5 lesson from Day 19: weight sharing makes convolutions cheap, and dense layers cost the most.

### Why 3×3 Convolutions

VGG stacks small kernels. Two 3×3 layers see a 5×5 region, and three see 7×7, with fewer parameters than one large kernel. For $C$ channels in and out:

$$\underbrace{3 \cdot (3^2 C^2)}_{\text{three } 3\times3} = 27C^2 \qquad \text{vs} \qquad \underbrace{7^2 C^2}_{\text{one } 7\times7} = 49C^2$$

The stack also adds three ReLUs in place of one, so it learns a more flexible function.

### Receptive Field by Layer

The receptive field grows through each convolution by $(k-1) \cdot j$, where $j$ is the cumulative stride, and $j$ doubles after every pool. Computed layer by layer:

| Layer | Index | Output shape | Receptive field (pixels) |
|---|---|---|---|
| block1_conv1 | 1 | 224×224×64 | 3 |
| block1_conv2 | 2 | 224×224×64 | 5 |
| block2_conv2 | 5 | 112×112×128 | 14 |
| block3_conv3 | 9 | 56×56×256 | 40 |
| block4_conv3 | 13 | 28×28×512 | 92 |
| block5_conv3 | 17 | 14×14×512 | 196 |

A unit in `block1_conv1` sees a 3×3 patch. A unit in `block5_conv3` sees 196 of the 224 pixels along each side, close to the whole image. This growth is what lets deep layers respond to parts and objects while early layers respond to edges.

## Part 1: Inspecting the Filters

List every convolution layer and the shape of its weight tensor:

```python
for i in range((len(model.layers))):
    if 'conv' not in model.layers[i].name:
        continue
    filter, bias = model.layers[i].get_weights()
    print(f"Layer {i} : {model.layers[i].name} {filter.shape}")
```

```text
Layer 1 : block1_conv1 (3, 3, 3, 64)
Layer 2 : block1_conv2 (3, 3, 64, 64)
Layer 4 : block2_conv1 (3, 3, 64, 128)
Layer 5 : block2_conv2 (3, 3, 128, 128)
Layer 7 : block3_conv1 (3, 3, 128, 256)
Layer 8 : block3_conv2 (3, 3, 256, 256)
Layer 9 : block3_conv3 (3, 3, 256, 256)
Layer 11 : block4_conv1 (3, 3, 256, 512)
Layer 12 : block4_conv2 (3, 3, 512, 512)
Layer 13 : block4_conv3 (3, 3, 512, 512)
Layer 15 : block5_conv1 (3, 3, 512, 512)
Layer 16 : block5_conv2 (3, 3, 512, 512)
Layer 17 : block5_conv3 (3, 3, 512, 512)
```

The shape is `(height, width, input channels, number of filters)`. In `block1_conv1`, `(3, 3, 3, 64)` means 64 filters, each holding one 3×3 kernel for each of the 3 input channels. The index gaps (3, 6, 10, 14) are the pooling layers, which have no weights. Index 0 is the input layer, so `model.layers[1]` is the first convolution.

Pull out the first convolution's weights and rescale them to [0, 1] so they display as grayscale images:

```python
filters, bias = model.layers[1].get_weights()
```

```python
# Normalizing the Weights
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
```

$$\hat{w} = \frac{w - w_{\min}}{w_{\max} - w_{\min}}$$

The minimum and maximum come from all 1,728 weights at once ($3 \cdot 3 \cdot 3 \cdot 64$), so each filter's brightness is relative to the global extremes, not to its own range.

Plot the first 6 filters, one row per filter and one column per input channel:

```python
import matplotlib.pyplot as plt

n_filters = 6
ix = 1
fig = plt.figure(figsize=(15, 10))
plt.title("Some Filters")
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):
        plt.subplot(n_filters, 3, ix)
        plt.imshow(f[:,:,j], cmap='gray')
        ix+=1

plt.show()
```

### What the Filters Show

- **Most filters ignore color.** In rows 1, 3, 4, and 5, the three channel kernels look the same. Those filters respond to brightness patterns regardless of color. Row 3 is bright on the left and dark on the right, a vertical-edge detector. Row 1 runs bright in the top-left to dark in the bottom-right, a diagonal-edge detector.
- **Some filters are color-sensitive.** In rows 2 and 6, the three channel kernels differ from one another, so those filters weigh the color channels differently. Row 2's third kernel looks unlike its first two.
- **Channel order.** Keras VGG16 weights expect BGR input after `preprocess_input`, so the three columns correspond to blue, green, red.
- **The learned filters match classical hand-built ones.** Edge and gradient kernels like Sobel appear in the network without anyone designing them. Training found them because edges are the cheapest useful feature in natural images.

Plot detail: `plt.title(...)` ran before any subplot, which creates an empty extra axes. That is why the output reports 19 axes for 18 filter images, and the title sits on a blank axes behind the grid. `fig.suptitle("Some Filters")` places it correctly.

## Part 2: Feature Maps

A filter is a set of weights. A **feature map** is what that filter produces when it slides over one specific image:

$$A_k^{(l)} = \mathrm{ReLU}\!\Big(\sum_{c} W_{k,c}^{(l)} * A_c^{(l-1)} + b_k^{(l)}\Big)$$

Layer $l$ has one map per filter $k$. Bright pixels mark positions where that filter's pattern matched strongly, and black pixels mark no match (ReLU sets negative responses to zero).

### Extract the First Layer's Output

Build a model that stops at the first convolution, then prepare the image with the same preprocessing VGG16 saw in training:

```python
from keras import Model
model = Model(inputs = model.inputs, outputs = model.layers[1].output)

from keras.preprocessing import image

img = image.load_img("image.png", target_size=(224, 224))

img = image.img_to_array(img)

img = tf.expand_dims(img, axis = 0)

img = preprocess_input(img)
```

Two details matter here:

- **The sub-model trick.** `Model(inputs, outputs=layer.output)` reuses the pretrained weights and returns the intermediate activation. No retraining is involved. The variable name `model` now points to this truncated model, not the full VGG16.
- **`preprocess_input` is required.** For VGG16 it converts RGB to BGR and subtracts the ImageNet channel means (103.939, 116.779, 123.68), with no division by 255. A pretrained network expects the exact preprocessing used in its training. Skipping it raises no error and degrades every activation downstream.

### Plot the 64 Maps

```python
features = model.predict(img)

fig = plt.figure(figsize=(20, 15))
for i in range(1, features.shape[3] + 1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1], cmap = 'gray')
plt.show()
```

The output has shape `(1, 224, 224, 64)`: one 224×224 map per filter.

### What the First-Layer Maps Show

Your note: some of the feature maps show the cat's eyes and nose. The grid agrees, with two details worth adding.

- **Many maps trace edges.** Eye outlines, whisker lines, and fur strokes appear as thin bright lines on a dark background, the edge detectors seen in Part 1.
- **Some maps copy the image.** A few show a bright, full copy of the face, filters that respond to overall intensity.
- **Several maps light up the eyes and nose as solid bright regions.** At this depth the filters know nothing about cats. Eyes and nose hold the sharpest contrast on the face, so they produce the strongest edge and intensity responses. The layer detects contrast. The cat shows up as a side effect of where the contrast sits.
- **One map is almost black.** Its filter found nothing to match in this image.

## Part 3: Feature Maps Across Depth

Choose the last convolution of each block (indices 2, 5, 9, 13, 17) and build one model that returns all five outputs:

```python
model2 = VGG16()

layer_index = [2,5,9,13,17]
outputs = [model2.layers[i].output for i in layer_index]

model3 = Model(inputs = model2.inputs, outputs = outputs)
```

Run the image once and plot each layer's maps:

```python
feature_maps = model3.predict(img)

for i, fmap in zip(layer_index, feature_maps):
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Layer {i}", fontsize = 20)
    for j in range(1, features.shape[3]+1):
        plt.subplot(8,8,j)
        plt.imshow(fmap[0,:,:,j-1], cmap = 'gray')

plt.show()
```

### Results

| Index | Layer | Map size | Channels | What the 64 plotted maps show |
|---|---|---|---|---|
| 2 | block1_conv2 | 224×224 | 64 | Edges, whiskers, and bright eye and nose regions, much like layer 1 with more variety |
| 5 | block2_conv2 | 112×112 | 128 | Finer structure: oriented fur strokes, eye outlines, dotted whisker pads |
| 9 | block3_conv3 | 56×56 | 256 | Textures and parts: oriented fur patches, eye rings, nose and mouth blobs |
| 13 | block4_conv3 | 28×28 | 512 | Sparse part-level patterns: pairs of eye blobs, curved mouth and nose lines |
| 17 | block5_conv3 | 14×14 | 512 | Coarse, blocky, sparse activations over the eye and mouth regions |

Three trends run through the depth sequence.

**1. Resolution falls.** The maps shrink from 224×224 to 14×14, so one pixel in layer 17 covers a 16×16 patch of the input. The layers lose "where exactly" and gain "what kind."

**2. Abstraction rises.** Early maps look like the photo with the edges highlighted. By layer 9 the maps stop resembling the photo and show fur texture and part shapes. By layer 13 they flag a pair of eyes or a curved mouth line in one blob. This is the hierarchy from the receptive-field table: edges, then textures, then parts.

**3. Sparsity rises.** At layer 17, about 16 of the 64 plotted maps look completely black by my count from the image, and several more hold one or two lit pixels. A deep filter matches a specific pattern, and one cat face contains few of the patterns the network learned. A black map for this image does not mean a dead filter, because the same filter can fire on other images. A filter counts as dead only if it stays black across the whole dataset (the dying-ReLU case from Day 11).

One more feature appears at layer 9: a map that shows only a bright rectangular frame and nothing inside. That response comes from the zero padding at the image border, which creates a sharp edge the filter can detect.

### Reading the Grids

- **Each map has its own brightness scale.** `plt.imshow` stretches every map to its own minimum and maximum, so a weak response and a strong one can look equally bright. To compare maps, pass the same `vmin` and `vmax` to all of them.
- **The grids are one image.** Everything above describes how VGG16 responds to this photo. Feature maps show activations on an input, and they do not show what a filter looks for in general. Gradient ascent on the input and Grad-CAM answer those questions.

## Putting the Two Halves Together

The filter and feature-map experiments explain why the pretrained model in the first section works. The first layers learn edge and contrast detectors, which any image contains. The middle layers learn textures and shapes that cover fur, metal, and skin. Only the last layers learn parts specific to ImageNet classes. Reusing the early and middle layers transfers real, general knowledge, and the late layers are the part to retrain.

Feature maps also work as a debugging tool. If every map in the first layer comes out black or looks like noise, the likely cause is wrong preprocessing, such as a missing `preprocess_input` or a mismatched channel order. Plotting five layers on one known image takes a minute and catches that class of bug before any training run.

## One-Line Takeaways

- Pretrained models solve data hunger and training time: someone else already paid for both.
- VGG16 has 138,357,544 parameters, and 89% of them sit in the three dense layers.
- Early convolution filters are edge and color detectors, and the first layer learns them on its own.
- A feature map is one filter's response to one image, and bright pixels mark where its pattern matched.
- Going deeper shrinks the maps, raises abstraction, and makes activations sparse.
- Always apply the model's own `preprocess_input`, because skipping it gives no error.
- Plotted feature maps scale each map separately, and the loop in Part 3 shows only the first 64 channels per layer.