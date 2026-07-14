# Part 2 — How a Neural Network Trains

> **Course:** NVIDIA DLI — Fundamentals of Deep Learning
> **Materials covered:** Lecture slides (Part 2) + Lab `02_asl.ipynb`
> **Previous:** [Part 1 — Introduction to Deep Learning](01-introduction-to-deep-learning.md)

---

## 📖 Section A — Lecture Notes

### 1. Recap: What Actually Happened in the MNIST Lab?

Stripping the last lab down to its skeleton, we did four things:

1. **Loaded and visualized** the data.
2. **Reshaped and normalized** it — each 28×28 image became an array of **784 values**, divided by 255 (the max pixel value) so everything sits between 0 and 1.
3. **Made the targets categorical** — instead of the label being the integer `3`, it becomes a vector of zeros with a `1` at index 3 (*one-hot encoding*):

   ```
   0 → [1,0,0,0,0,0,0,0,0,0]
   1 → [0,1,0,0,0,0,0,0,0,0]
   3 → [0,0,0,1,0,0,0,0,0,0]
   ```

   This matches the model's 10 output neurons — one probability per class — which is a form the model trains against much better than a raw number.
4. **Created, compiled, and trained** the model.

#### How big was that "simple" model?

Every neuron in a layer connects to every neuron in the next (PyTorch calls this a **Linear** layer; Keras calls the same thing **Dense**). Counting connections:

```
784×512 + 512×512 + 512×10  ≈  668,672 weights
```

Over half a million connections in a four-layer toy network — and the network's *knowledge* is stored entirely in those **weights**, analogous to the connection strengths between neurons in a brain. So the real question of this lecture: **how do the weights get updated?**

---

### 2. A Single Neuron = Linear Regression

To understand the whole network, shrink it to one neuron. The earliest use of a neuron was to fit a regression line — good old:

```
ŷ = m·x + b
```

- **m** (slope) is the neuron's **weight** for the input.
- **b** (intercept) is the neuron's **bias**.
- **ŷ** ("y-hat") is our *estimate* of the true value y.

**Regression** = using a continuous input to predict a continuous output (e.g., seconds a pot has been on the stove → water temperature).

**Toy dataset:** the points (1, 3) and (2, 5). Algebra tells us the perfect line is m = 2, b = 1. But the network won't use algebra — it will use **trial and error**, starting from *random* values (say m = −1, b = 5) and improving. This is the miniature version of everything a neural network does.

### 3. Measuring How Wrong We Are: the Loss Function

To improve, we need a score for "how bad is the current guess." Classic regression uses **Mean Squared Error (MSE)**: for each point, take the difference between prediction and truth, **square it** (two reasons: errors stay positive so they can't cancel out, and big mistakes get punished disproportionately), then average.

```python
data = [(1, 3), (2, 5)]

def get_rmse(data, m, b):
    n = len(data)
    squared_error = 0
    for x, y in data:
        y_hat = m * x + b            # prediction
        squared_error += (y - y_hat) ** 2
    mse = squared_error / n         # mean squared error
    return mse ** 0.5               # root → back to original units
```

The function we choose to evaluate the model — MSE here — is called the **loss function** (or cost function).

### 4. The Loss Curve and Gradient Descent

Now imagine plotting the loss for *every possible* combination of m and b. For our toy problem the surface is a **bowl** (a parabolic ellipse) that bottoms out exactly at m = 2, b = 1. Training = getting to the bottom of the bowl.

A human can eyeball the bottom. The computer can't "see" the surface — and computers are surprisingly **bad at calculus** (they can't just solve for the global minimum analytically in general). What they *are* good at is computing the **slope at their current position**. So we let the computer cheat:

1. **Compute the gradient** — "gradient" is just the word for a slope in multiple dimensions. It tells us in which direction (some ratio of changing b vs changing m) the loss decreases fastest. One simple way to get it: evaluate the function at two extremely close points and measure the slope between them.
2. **Take a step in that direction.** How big a step? That's the **learning rate (λ)** — a value we choose:
   - **Too large** → you overshoot and can end up *farther* from the target.
   - **Too small** → training takes forever, and on complex problems you can get **trapped in a local minimum** (a dip that isn't the true bottom).
3. **Repeat** — new position, new gradient, new step — until the slope is ~0 for every parameter, meaning we've reached a minimum.

This walk downhill is called **gradient descent**. It's the beating heart of all deep learning.

**Vocabulary checkpoint:**
- **Step** — take a batch (or the full dataset), compute the gradient, update the parameters.
- **Epoch** — one full pass through the dataset. Historically gradients were computed on the whole dataset at once; today it's more efficient to use **batches**.

### 5. Optimizers

Choosing and adjusting the learning rate by hand is painful, so frameworks ship **optimizers** that adapt it automatically: **Adam, Adagrad, RMSprop, SGD**…

The intuition for **Adam** (*Adaptive momentum*): treat the loss surface as a mountain and your position as a **marble**. Dropped from the top, the marble picks up speed — enough momentum to *jump over small trenches* (local minima) and hopefully settle in a deeper, better minimum. That's why we used `Adam(model.parameters())` in the lab without ever setting a learning rate ourselves.

---

### 6. From One Neuron to a Network

Two moves scale a lonely neuron into deep learning:

1. **More inputs.** Instead of one x with one slope m, take x₁, x₂, x₃… each with its own weight w₁, w₂, w₃… The math barely changes — you just compute a gradient for each new weight, exactly as before.
2. **Chain neurons.** Feed the output of one neuron into the inputs of others (any wiring is fine as long as there's no loop). Congratulations — that's officially a deep neural network. During gradient descent, the error computed at a *later* neuron feeds into the weight updates of the *earlier* neurons connected to it — this flowing of error backwards through the chain is **backpropagation**. Frameworks automate the math (`.backward()` in PyTorch did this for us).

### 7. Activation Functions — Why Lines Aren't Enough

⚠️ **The gotcha:** if every neuron only computes a weighted sum (a linear function), then any stack of them — however deep — collapses into… another linear function. Adding lines gives you a line. The whole network would be a waste; it could never capture complex shapes.

**The fix:** pass each neuron's linear output through a **non-linear function** — the *activation function*. Computers love lines (fast to compute, easy to differentiate), so we keep the line and just warp its output:

| Activation | What it does | Why it's useful |
|---|---|---|
| **Linear** (none) | `ŷ = wx + b` passes straight through | Fine for a final regression output |
| **ReLU** (Rectified Linear Unit) | If the output is negative → set it to 0 | Dirt-cheap non-linearity; the workhorse of hidden layers |
| **Sigmoid** | Warps the line into an S-shape squeezed between 0 and 1 | Perfect for outputting a **probability** |

In code, they're tiny:

```python
y_hat = w*x + b                 # Linear
y_hat = y_hat * (y_hat > 0)     # ReLU: zero out negatives

linear = w*x + b                # Sigmoid: squash to (0, 1)
y_hat = 1 / (1 + np.exp(-linear))
```

**Why non-linearity pays off — the two-waves example:** suppose your data looks like the sum of two sine waves. Give the first layer a sinusoidal activation and keep the last layer linear (so it can add the waves), and with enough data the network will discover the parameters of each wave *on its own*. The broader lesson: understanding the shape of your data lets you build more efficient models — saving time, compute, and money.

---

### 8. Overfitting — Why Not Just Build a Giant Network?

If more layers and neurons can capture more complex shapes, why not throw a massive network at everything? Beyond wasted compute and slow training, there's a deeper trap.

**The two-trendlines example:** fit two models to a sample of data —

| | Wiggly complex model | Simple straight-ish model |
|---|---|---|
| MSE on the data you fit | **0.0000** (perfect!) | 0.0113 |
| MSE on a *fresh* sample | 0.0308 | **0.0062** |

The "perfect" model was tracing the noise in that particular sample. On new data, the *simpler* model **generalizes** better. A model performing great on training data but poorly on new data has **overfit** — it *memorized* the deck of flashcards instead of learning the concepts.

**How to detect it:** this is exactly what the validation set is for. Plot loss per epoch:

- **Healthy:** training loss and validation loss both fall (validation a bit higher — that's expected).
- **Overfitting:** training loss keeps falling while validation loss bottoms out and then **climbs**. The gap between the curves is the memorization.

Picking the right model complexity for the data is a core judgment call of the data scientist. (Concrete remedies come in Parts 3–4.)

---

### 9. From Regression to Classification

Revisiting the MNIST model diagram with our new vocabulary: 784 inputs → Linear(512) + **ReLU** → Linear(512) + **ReLU** → Linear(10) with **no activation**. Why is the output layer bare?

#### 9.1 Softmax — hiding inside CrossEntropyLoss

During **prediction**, raw output scores are fine: the biggest number wins (`argmax`). During **training**, `CrossEntropyLoss` implicitly applies a **Softmax** to the outputs: conceptually, each output neuron goes through a sigmoid to become a probability, then the probabilities are **normalized so they sum to 100%**. Now the model's guess is a proper probability distribution over the 10 classes — which is what cross entropy knows how to grade.

That's why the output layer has no ReLU/sigmoid of its own: the loss function supplies the softmax during training.

#### 9.2 Why not just use MSE for classification?

Imagine points colored red or blue. Regression wants a line *through* points; classification wants a line that *separates* them, plus a confidence for each prediction. You could hack numbers onto colors and use RMSE, but there's a loss designed for the job.

#### 9.3 Cross entropy — "infinite error for misplaced confidence"

The design logic, before any math:

- 100% confident it's blue, and it **is** blue → **zero loss**. Perfect.
- 100% confident it's blue, and it's **not** → **infinite loss**. Punished for hubris.

Loss grows slowly for small mistakes and explodes toward infinity as confident predictions turn out wrong. The logarithm is what supplies the "infinite" part (log(0) = −∞):

```python
def cross_entropy(y_hat, y_actual):
    """Infinite error for misplaced confidence."""
    loss = log(y_hat) if y_actual else log(1 - y_hat)
    return -1 * loss
```

The full binary cross-entropy formula looks scary, but there's a trick: it's two halves, and **one half cancels to zero** depending on whether the target is true or false — leaving exactly the code above. *Categorical* cross entropy is the same idea extended to many classes, which is what `nn.CrossEntropyLoss` computes for our 10 (or 24) categories.

---
---

## 🧪 Section B — Lab: ASL Image Classification (`02_asl.ipynb`)

Same pipeline as MNIST — but now **you** build each piece, with a new dataset, and the lab ends with a deliberate lesson.

### 1. The Dataset: American Sign Language Alphabet

- Images of hands signing letters, **28 × 28 grayscale** — same shape as MNIST.
- **24 static letters**: `j` and `z` are excluded because they require *movement*, which a single image can't capture.
- Source: **Kaggle** — worth knowing as a general resource: datasets, notebook "kernels", and competitions for practicing DL.

> Small quirk to notice: the notebook sets `n_classes = 26` (full alphabet) even though only 24 letters have data. The model just learns that two output neurons never fire — harmless here.

### 2. Loading Custom Data with Pandas

MNIST came pre-packaged in torchvision; the real world doesn't. This dataset arrives as **CSV** files where *each row is one image*: a `label` column + 784 pixel columns.

```python
import pandas as pd

train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")
train_df.head()   # peek at the first rows
```

Separate labels from pixels — `pop` removes the column from the DataFrame *and* hands it to you:

```python
y_train = train_df.pop('label')   # labels out...
x_train = train_df.values         # ...pixels (as a NumPy array) remain
```

Resulting shapes: **27,455 training images**, **7,172 validation images**, each 784 pixels.

To visualize one, reshape the flat 784-vector back into a 2-D image: `row.reshape(28, 28)` → `plt.imshow(image, cmap='gray')`.

### 3. Exercise 1 — Normalize

Pixels arrive as integers `[0, 255]`; the model wants floats `[0, 1]`. With `ToTensor` unavailable (we're not starting from PIL images), do it the direct way:

```python
x_train = train_df.values / 255
x_valid = valid_df.values / 255
```

Same normalization as MNIST — just manual this time.

### 4. Custom PyTorch Datasets

To plug arbitrary data into a DataLoader, subclass `Dataset` and implement three methods:

```python
class MyDataset(Dataset):
    def __init__(self, x_df, y_df):                     # runs once at creation
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):                         # returns one (x, y) pair
        return self.xs[idx], self.ys[idx]

    def __len__(self):                                  # dataset size
        return len(self.xs)
```

💡 **Performance note:** in the MNIST lab we moved each batch to the GPU inside the training loop (`x.to(device)` per batch). Here the dataset is small enough to **park entirely on the GPU once, in `__init__`** — so the training loop no longer needs any `.to(device)` calls. Two valid strategies; choose by dataset size vs GPU memory.

A custom Dataset then behaves exactly like a prebuilt one:

```python
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
```

**Sanity-check your DataLoader** before training — draw one hand from the deck:

```python
batch = next(iter(train_loader))
batch[0].shape   # x: torch.Size([32, 784])
batch[1].shape   # y: torch.Size([32])
```

Run it a few times — with `shuffle=True` the values change each draw. Checking shapes like this catches most data bugs early.

### 5. Exercise 2 — Build the Model

Same architecture as MNIST, different output size:

```python
input_size = 28 * 28
n_classes = 26

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),
    nn.Linear(512, 512),         # Hidden
    nn.ReLU(),
    nn.Linear(512, n_classes)    # Output
)
model = torch.compile(model.to(device))   # GPU + compile in one line

loss_function = nn.CrossEntropyLoss()     # same as MNIST — still categorizing
optimizer = Adam(model.parameters())
```

### 6. Exercise 3 — Train and Validate

Same functions as Part 1 with one difference: **no `x.to(device)` in the loop**, because the Dataset already lives on the GPU. The six training steps per batch, now stated explicitly by the course:

1. Get an `output` prediction from the model
2. Zero the gradient — `optimizer.zero_grad()`
3. Calculate loss — `loss_function(output, y)`
4. Compute the gradient — `batch_loss.backward()`
5. Update the parameters — `optimizer.step()`
6. Accumulate `loss` and `accuracy` totals for reporting

The accuracy-helper exercise (fill in the FIXMEs) reinforces what each argument does:

```python
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)         # model scores → predicted class
    correct = pred.eq(y.view_as(pred)).sum().item()   # compare with true labels
    return correct / N                                # fraction of the whole dataset
```

Then train for **20 epochs**.

### 7. The Punchline: What Happened?

Training accuracy climbs very high — but **validation accuracy lags noticeably behind**. The model is *memorizing* the training deck rather than gaining a robust, general understanding of hand shapes.

This is **overfitting**, live and on purpose — the exact phenomenon Section A predicted with the two-trendlines example. ASL hands vary more than centered digits do, and our fully-connected network treats each pixel independently, so it latches onto training-set specifics. The fix (convolutional networks, data augmentation) is the subject of Parts 3 and 4.

### 8. Housekeeping

As always, shut down the kernel to free GPU memory before the next notebook:

```python
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

---

## 🔑 Key Takeaways

1. **A neuron is a tiny regression:** `ŷ = wx + b`. Weights = slopes, bias = intercept. A network's entire knowledge lives in its weights (~668k of them even in our toy MNIST model).
2. **Training = gradient descent:** compute the loss, find the gradient (multi-dimensional slope), step downhill, repeat. Computers can't "see" the loss bowl — they feel their way down with local slopes.
3. **Learning rate is a balancing act:** too big overshoots the minimum; too small crawls and can strand you in a local minimum. **Optimizers** (Adam = marble with momentum rolling down a mountain) manage it for you.
4. **Step vs epoch:** a step = one batch → gradient → update; an epoch = one full pass through the dataset.
5. **Backpropagation** chains gradient descent through connected neurons: later errors inform earlier weights. Frameworks do the math (`.backward()`).
6. **No activation functions → no deep learning.** Sums of linear functions are still linear. **ReLU** (zero out negatives) adds cheap non-linearity in hidden layers; **Sigmoid** squashes to (0,1) for probabilities.
7. **Softmax lives inside CrossEntropyLoss:** during training the raw output scores become normalized probabilities summing to 100% — that's why the output layer has no activation of its own.
8. **Cross entropy = infinite error for misplaced confidence:** near-zero loss for correct confident predictions, exploding loss for wrong confident ones (courtesy of the logarithm).
9. **Overfitting = memorization:** great training metrics, poor validation metrics. A simpler model that generalizes beats a complex model that's perfect on data it has already seen. Watch the *gap* between the two curves.
10. **Custom data pipeline pattern:** pandas `read_csv` → `pop` the label → normalize (`/255`) → custom `Dataset` (`__init__`, `__getitem__`, `__len__`) → `DataLoader`. Small datasets can move to the GPU once in `__init__` instead of per-batch.
