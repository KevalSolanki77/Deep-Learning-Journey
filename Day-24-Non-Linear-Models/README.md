# Day 24 - Non-linear Models (Keras Functional API)

## Terminology

"Non-linear" in this entry means non-linear **topology**: a graph of layers. It does not mean non-linear activations like ReLU.

- A `Sequential` model is a straight chain: one input, a line of layers, one output.
- Anything with branches, merges, multiple inputs, or multiple outputs is a graph.
- The Functional API builds graphs.

## Where Sequential Breaks

`Sequential` assumes one input and one output with layers stacked in a single line. It cannot express these cases:

| Limitation | Why it matters | Example |
|---|---|---|
| Multiple inputs | Cannot feed an image and tabular data into the same model | Product photo + product metadata |
| Multiple outputs | Cannot predict two things at once (multi-task learning) | Age and gender from one face |
| Skip / residual connections | Cannot add a layer's input to its output, the trick that makes deep nets trainable | ResNet |
| Shared layers | Cannot reuse the same weights on two inputs | Siamese networks |
| Branching | Cannot run parallel paths and merge them | Inception-style blocks |

## The Functional API

The core idea: **layers are functions you call on tensors.** You wire the graph yourself.

```python
x = Layer(...)(x)               # call a layer on a tensor, get a tensor back
model = Model(inputs=..., outputs=...)   # the graph is everything between them
```

Each call adds one node. `Input(...)` creates the entry tensor. `Model(inputs, outputs)` collects every layer on the path from the inputs to the outputs. The `Connected to` column in `model.summary()` lists each layer's parents, which is the graph written as a table.

The model is a directed acyclic graph. Each node $v$ computes its output from its parents' outputs:

$$h_v = f_v\big(h_{u_1}, h_{u_2}, \dots\big) \qquad u_i \in \mathrm{parents}(v)$$

A `Sequential` model is the special case where every node has one parent.

## The Five Patterns

### 1. Multi-Input (Tabular + Image)

Build one arm per input, flatten each arm to a vector, and join the vectors with `concatenate`. Experiment 2 and Experiment 3 below use this pattern.

`Concatenate` joins along one axis and needs matching sizes on every other axis, so the two arms can have different feature counts. A `Dense` layer after the join then mixes the two sources:

$$\text{params}_{\text{Dense after concat}} = (n_{img} + n_{meta}) \cdot n_{units} + n_{units}$$

### 2. Skip Connection (the ResNet Idea)

A residual block computes a transformation $\mathcal{F}$ and adds the block's input back to its output:

$$y = \mathcal{F}(x) + x$$

The gradient shows why this helps deep networks. Differentiate through the block:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)$$

The `1` is the skip path. Even when $\partial \mathcal{F} / \partial x$ shrinks toward zero (the vanishing gradient problem from Day 5), the gradient still flows through the identity term. Stack $L$ blocks and the signal at block $l$ includes a direct, unmultiplied path from the loss:

$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i)$$

That unbroken path lets networks reach 100+ layers.

Reference snippet (not in the notebook; run and checked):

```python
from keras import layers
from keras.models import Model

inp = layers.Input((32, 32, 64))

x = layers.Conv2D(64, 3, padding='same')(inp)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, inp])      # skip connection: F(x) + x
x = layers.ReLU()(x)

block = Model(inp, x)           # 74,368 parameters, output (None, 32, 32, 64)
```

`Add` needs identical shapes. When the block changes the channel count, project the skip path with a 1×1 convolution so both paths match:

```python
shortcut = layers.Conv2D(128, 1)(inp)    # 64 channels -> 128 channels
y = layers.Add()([y, shortcut])
```

Without the projection, Keras raises `ValueError: Inputs have incompatible shapes. Received shapes (32, 32, 128) and (32, 32, 64)`.

### 3. Multi-Output

One shared trunk feeds several heads, each with its own loss. Experiment 1 and Experiment 3 use this pattern. The training objective is a weighted sum:

$$\mathcal{L} = \sum_k w_k \, \mathcal{L}_k$$

Every head sends gradient back into the shared trunk, so the trunk update is the weighted sum of what each task asks for:

$$\frac{\partial \mathcal{L}}{\partial \theta_{trunk}} = \sum_k w_k \, \frac{\partial \mathcal{L}_k}{\partial \theta_{trunk}}$$

When one loss has a much larger scale than another, its term dominates that sum and the other task moves the trunk very little. The weights $w_k$ fix that.

### 4. Shared Layers (Siamese)

Define the layer once, then call it on two inputs. Both branches use identical weights, so the model learns one embedding space where similar inputs land close together.

Reference snippet (not in the notebook; run and checked):

```python
import keras
from keras import layers
from keras.models import Model

shared = layers.Dense(64, activation='relu', name='shared_encoder')   # one instance

a = layers.Input((100,), name='left')
b = layers.Input((100,), name='right')

ea = shared(a)      # same weights
eb = shared(b)      # same weights

diff = layers.Lambda(lambda t: keras.ops.abs(t[0] - t[1]))([ea, eb])
out = layers.Dense(1, activation='sigmoid', name='same_person')(diff)

siamese = Model([a, b], out)    # 6,529 parameters
```

The parameter count proves the sharing: the encoder costs $100 \cdot 64 + 64 = 6{,}464$ once, plus the output layer's $64 + 1 = 65$, which totals 6,529. Two separate `Dense(64)` instances on the same inputs would cost 12,928.

During backpropagation the shared weights receive the sum of the gradients from both branches:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W}\bigg|_{\text{left use}} + \frac{\partial L}{\partial W}\bigg|_{\text{right use}}$$

### 5. Branching and Merging

Run parallel paths on the same input and merge them. Inception blocks run 1×1, 3×3, and 5×5 convolutions side by side and concatenate the results. Wide & Deep models run a linear path beside a deep path and merge both at the output. The Functional API handles both with the same two tools: call several layers on one tensor, then `concatenate` or `Add` the results.

## The Experiments

The notebook builds three models of increasing complexity. It calls `model.summary()` and `plot_model()` on each. It never trains any of them, so this section has architecture and parameter counts and no accuracy numbers.

```python
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Rescaling,
    ReLU,
    Flatten,
    Dropout,
    concatenate
)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import plot_model
```

### Experiment 1: One Input, Two Outputs

Input: a human face image (256×256×3). Outputs: **age** (regression) and **gender** (binary classification).

```python
# Input Image
input_layer = Input(shape=(256, 256, 3))

# Normalization
x = Rescaling(1./255)(input_layer)

# Convolutional Blocks
x = Conv2D(32, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

# Flatten feature map for Dense layers
x = Flatten()(x)

# Age Output
y = Dense(128, activation='relu')(x)
age_output = Dense(1, activation='linear', name='age_output')(y)

# Gender Output
z = Dense(128, activation='relu')(x)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(z)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
model.summary()
```

The conv trunk is shared. After `Flatten`, the graph splits into two heads, each with its own `Dense(128)`. The output activations match the task: `linear` for a number (age) and `sigmoid` for a probability (gender).

**Shape trace.** Each block is a stride-2 convolution (2×2 kernel, `valid`) followed by a 2×2 max-pool, so each block halves the map twice:

| Stage | Output shape |
|---|---|
| Input / Rescaling | 256×256×3 |
| Block 1: conv, pool | 128×128×32, then 64×64×32 |
| Block 2: conv, pool | 32×32×64, then 16×16×64 |
| Block 3: conv, pool | 8×8×128, then 4×4×128 |
| Flatten | 2,048 |
| Two heads | 1 (age) and 1 (gender) |

**Parameters**, with the arithmetic that reproduces the summary's 567,266:

| Layer | Arithmetic | Parameters |
|---|---|---|
| Conv 32 | 2·2·3·32 + 32 | 416 |
| BatchNorm (32) | 4 × 32 | 128 |
| Conv 64 | 2·2·32·64 + 64 | 8,256 |
| BatchNorm (64) | 4 × 64 | 256 |
| Conv 128 | 2·2·64·128 + 128 | 32,896 |
| BatchNorm (128) | 4 × 128 | 512 |
| Age Dense 128 | 2,048·128 + 128 | 262,272 |
| Gender Dense 128 | 2,048·128 + 128 | 262,272 |
| Age output | 128 + 1 | 129 |
| Gender output | 128 + 1 | 129 |

Total **567,266**: 566,818 trainable and 448 non-trainable (the BatchNorm running means and variances, $2 \times (32 + 64 + 128)$). The two head `Dense(128)` layers hold 92.5% of the model, the same pattern as LeNet-5's C5 (Day 19) and VGG16's classifier (Day 22).

**Compile with a loss per output and weights between them:**

```python
model.compile(
    optimizer="adam",
    loss={"age_output": "mse", "gender_output": "binary_crossentropy"},
    loss_weights={"age_output": 0.01, "gender_output": 1.0}
)

plot_model(model, show_shapes = True)
```

The loss dictionary maps by output **name**, which is why the two output layers carry `name='age_output'` and `name='gender_output'`.

**Why the 0.01.** Age error in squared years grows fast: a 10-year miss costs $10^2 = 100$ under MSE. Binary cross-entropy for gender starts near 0.69 and falls below 1. Left unweighted, the age term would drive most of the shared trunk's gradient. Weighting age by 0.01 puts a 10-year miss at 1.0, the same size as the gender loss. The weights are a starting guess. Tune them, or scale the age target (for example, divide by 100), and watch both per-head losses.

### Experiment 2: Two Inputs, One Output

Inputs: a product image (224×224×3) and 10 numeric metadata fields. Output: the **price**.

```python
# Input Layers
img = Input(shape=(224, 224, 3))
data = Input(shape=(10,))

# Image Part
x = Rescaling(1./255)(img)
x = Conv2D(32, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, kernel_size=(2,2), strides=2, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D()(x)

# Correctly flatten the image features
x = Flatten()(x)

# Tabular Data Part
y = Dense(128)(data)
y = BatchNormalization()(y)
y = ReLU()(y)
y = Dropout(0.2)(y)

y = Dense(64)(y)
y = BatchNormalization()(y)
y = ReLU()(y)
y = Dropout(0.2)(y)

y = Dense(32)(y)
y = BatchNormalization()(y)
y = ReLU()(y)
y = Dropout(0.2)(y)

# Concatenation (both are now 2D tensors)
combined = concatenate([x, y])

# Combined NN
z = Dense(128, activation='relu')(combined)
price_output = Dense(1, activation='linear')(z)

model2 = Model(inputs=[img, data], outputs=price_output)
model2.summary()

plot_model(model2, show_shapes=True)
```

The graph has two arms that meet at `concatenate`. The image arm becomes a vector with `Flatten`, the tabular arm is already a vector, and `concatenate` needs 2D tensors of shape `(batch, features)` from both, which is what the comment in the code points at.

**Shape trace of the image arm.** The map is odd-sized after the last convolution, so the pool floors it:

| Stage | Output shape |
|---|---|
| Input / Rescaling | 224×224×3 |
| Block 1: conv, pool | 112×112×32, then 56×56×32 |
| Block 2: conv, pool | 28×28×64, then 14×14×64 |
| Block 3: conv, pool | 7×7×128, then 3×3×128 (pool drops the odd row and column) |
| Flatten | 1,152 |

The tabular arm outputs 32 values, so `concatenate` produces $1{,}152 + 32 = 1{,}184$ features.

**Parameters**, reproducing the summary's 206,913:

| Part | Arithmetic | Parameters |
|---|---|---|
| Image convs | 416 + 8,256 + 32,896 | 41,568 |
| Image BatchNorms | 128 + 256 + 512 | 896 |
| Tabular Dense 128, 64, 32 | 1,408 + 8,256 + 2,080 | 11,744 |
| Tabular BatchNorms | 512 + 256 + 128 | 896 |
| Dense 128 after concat | 1,184·128 + 128 | 151,680 |
| Output | 128 + 1 | 129 |

Total **206,913**: 206,017 trainable and 896 non-trainable. The `Dense(128)` after the merge holds 73% of the model.

Both arms apply `BatchNormalization` right after a `Dense` or `Conv2D` that keeps its default bias. Day 13 covered why that bias is redundant: BatchNorm subtracts the batch mean, and its `β` supplies the shift. Setting `use_bias=False` on those layers saves parameters without changing what the model can learn.

### Experiment 3: Two Inputs, Two Outputs, Pretrained Backbone

Inputs: product image (processed by a frozen VGG16) and 10 metadata fields. Outputs: **category** (15-class classification) and **price** (regression).

```python
# 1. Inputs
img_input = Input(shape=(224, 224, 3), name='image_input')
meta_input = Input(shape=(10,), name='metadata_input')

# 2. Image Feature Extraction using VGG16 Conv Base
# Preprocess inputs for VGG16 compatibility
x_img = preprocess_input(img_input)

# Load pre-trained VGG16 without the classifier top layers
vgg_base = VGG16(include_top=False, weights='imagenet', input_tensor=x_img)

# Freeze VGG16 convolutional layers to retain pre-trained features
for layer in vgg_base.layers:
    layer.trainable = False

# Flatten the extracted image feature maps
x_img_features = Flatten()(vgg_base.output)

# 3. Tabular Feature Extraction
x_meta = Dense(64)(meta_input)
x_meta = BatchNormalization()(x_meta)
x_meta = ReLU()(x_meta)
x_meta = Dropout(0.2)(x_meta)

# 4. Merging the Image & Tabular features
combined_features = concatenate([x_img_features, x_meta])

# Combined Representation Layer
x_shared = Dense(256, activation='relu')(combined_features)

# 5. Multi-Outputs
category_output = Dense(15, activation='softmax', name='category_output')(x_shared)
price_output = Dense(1, activation='linear', name='price_output')(x_shared)

# 6. Instantiate and Summarize Model
model3 = Model(inputs=[img_input, meta_input], outputs=[category_output, price_output])
model3.summary()
```

This model uses all three ideas at once: two inputs, a merge, and a shared representation (`x_shared`) feeding two heads.

Two details in how the graph gets built:

- **`preprocess_input` runs on a symbolic tensor.** Day 23 wrapped it in a `Lambda` layer. Here Keras records its operations straight into the graph, and `plot_model` shows them as a `GetItem` ×3 → `Stack` → `Add` chain at the top: split the three channels, reorder them (RGB to BGR), and add the negative ImageNet channel means. None of these nodes has parameters.
- **`input_tensor=x_img` puts VGG16's layers into the same graph.** In Day 23, `VGG16` sat inside the model as one nested `vgg16` block. Here its 13 convolutions and 5 pools appear as individual nodes in `model3`.

The plot also shows the shape of the merge: the image arm ends in `Flatten` at 25,088 values, the metadata arm ends at 64, and `Concatenate` outputs 25,152.

**Parameters**, reproducing the summary's 21,158,928:

| Part | Arithmetic | Parameters |
|---|---|---|
| VGG16 conv base (frozen) | 13 conv layers | 14,714,688 |
| Metadata Dense 64 | 10·64 + 64 | 704 |
| Metadata BatchNorm (64) | 4 × 64 | 256 |
| Dense 256 after concat | 25,152·256 + 256 | 6,439,168 |
| Category head | 256·15 + 15 | 3,855 |
| Price head | 256 + 1 | 257 |

Total **21,158,928**. Non-trainable: 14,714,816 (VGG16's 14,714,688 plus the metadata BatchNorm's 128 running statistics). Trainable: **6,444,112**.

**99.9% of the trainable parameters sit in one layer.** The `Dense(256)` after the merge costs 6,439,168 of the 6,444,112 trainable weights, because `Flatten` hands it 25,088 image features. Swapping `Flatten` for `GlobalAveragePooling2D` (Day 19) drops the image vector to 512 values and the layer to $576 \cdot 256 + 256 = 147{,}712$ parameters, which is 43.6 times fewer. That is the first change to try when the head overfits.

**Compile cell, as written in the notebook:**

```python
model.compile(
    optimizer="adam",
    loss={"category_output": "categorial_crossentropy", "price_output": "mae"},
    loss_weights={"price_output": 0.01, "category_output": 1.0}
)

plot_model(model3, show_shapes=True)
```

This cell has three problems. I rebuilt all three models and ran them to confirm.

1. **It compiles `model`, not `model3`.** `model` is Experiment 1's age and gender network. Keras accepts the call, so nothing fails here, but the first `fit` raises `ValueError: Expected keys ['age_output', 'gender_output'] in loss dict, but found loss.keys()=['category_output', 'price_output']`.
2. **`"categorial_crossentropy"` is a typo.** On the correct model, Keras accepts the compile call and `fit` fails with `ValueError: Could not interpret loss identifier: categorial_crossentropy`.
3. **The loss does not match integer labels.** `categorical_crossentropy` expects one-hot targets. Category labels stored as integers 0 to 14 need `sparse_categorical_crossentropy`.

Corrected:

```python
model3.compile(
    optimizer="adam",
    loss={"category_output": "sparse_categorical_crossentropy", "price_output": "mae"},
    loss_weights={"category_output": 1.0, "price_output": 0.01}
)
```

Named inputs and outputs then let `fit` map data by name:

```python
model3.fit(
    {"image_input": images, "metadata_input": meta},
    {"category_output": categories, "price_output": prices},
    epochs=10,
)
```

A small model built the same way (named inputs, named heads, loss and weight dictionaries) trained one epoch with this call and logged `loss`, `category_output_loss`, and `price_output_loss` as three entries. Those per-head losses are the numbers to watch when tuning `loss_weights`.

The 0.01 on price follows the same logic as Experiment 1: a price in the hundreds or thousands makes MAE far larger than cross-entropy, so it needs a small weight or a scaled target. MAE grows in proportion to the error, so a few extreme prices pull on the gradient less than they would under MSE.

## Gotchas

- **`Add` needs identical shapes.** Project a skip path that changes dimension with a `Dense` or 1×1 `Conv2D` first. `Concatenate` needs matching sizes only on the axes you are not joining.
- **Name your inputs and outputs.** Loss dictionaries and `fit({"numeric": ..., "image": ...})` map by name. Unnamed layers get auto-names (Experiment 2's inputs are `input_layer_5` and `input_layer_6`) that change between sessions, which breaks any dictionary written against them. Experiment 3 names everything, and that is the pattern to copy.
- **Call `model.summary()` and `plot_model()`.** Most wiring bugs show up in the graph at once: a missing connection, a wrong merge, an input that feeds nothing.
- **Multi-task losses fight each other.** A loss on a much larger scale (MSE in the thousands against cross-entropy near 1) dominates the gradients. Scale the targets or tune `loss_weights`.
- **Reusing a layer instance shares weights, by design.** For separate weights, create two instances (12,928 parameters against 6,464 in the Siamese example).
- **Flattening a big feature map into a `Dense` layer concentrates the parameters.** In Experiments 1, 2, and 3, the dense layer(s) right after the flatten or merge hold 92.5%, 73%, and 99.9% of the parameters (Experiment 1 splits its share across two head layers).

## Real-World Applications

| Architecture | Where it appears | Why the graph matters |
|---|---|---|
| ResNet | Almost every modern architecture, Transformers included, uses residuals | Skip connections let networks go 100+ layers deep without the gradient vanishing |
| Wide & Deep | Google Play recommendations | A wide linear path memorizes feature crosses while a deep path generalizes, and the two merge at the end |
| Self-driving perception | One shared backbone, multiple heads for lane detection, object detection, and depth | Multi-output saves compute compared with separate models |
| Face verification (Siamese) | Same encoder on two photos, compare the embeddings | Works with few examples per person |
| U-Net | Medical image segmentation | Skip connections carry fine spatial detail from encoder to decoder |
| E-commerce listings | A product photo plus attributes (brand, size, seller rating) predicting category and a suggested price | Same problem shape as Experiments 2 and 3 |

## One-Line Takeaways

- "Non-linear" here means graph topology, not activation functions.
- `Sequential` handles one input, one output, one chain. Everything else needs the Functional API.
- Layers are functions: call them on tensors and wire the graph with `Model(inputs, outputs)`.
- A skip connection computes $F(x) + x$, and the identity term keeps the gradient alive in deep networks.
- Multi-output models sum weighted losses, so scale or weight the heads to stop one loss from dominating.
- A shared layer instance shares weights, and its gradient is the sum over every use.
- The notebook builds three graphs and trains none of them. Its last compile cell targets the wrong model and has two loss errors.