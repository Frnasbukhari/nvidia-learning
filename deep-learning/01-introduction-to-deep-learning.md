# Part 1 — An Introduction to Deep Learning

> **Course:** NVIDIA DLI — Fundamentals of Deep Learning
> **Materials covered:** Lecture slides (Part 1) + Lab `01_mnist.ipynb`
> **Framework:** PyTorch 2

---

## 📖 Section A — Lecture Notes

### 1. Where Deep Learning Sits

Deep learning is a **subset of machine learning**, which is itself a subset of the wider field of **artificial intelligence**. The nesting matters:

```
Artificial Intelligence  ⊃  Machine Learning  ⊃  Deep Learning
```

- **AI** — any technique that lets computers mimic intelligent behavior (including hand-coded rules).
- **ML** — systems that *learn patterns from data* instead of being explicitly programmed.
- **DL** — machine learning using **deep neural networks**: models with many stacked ("hidden") layers. The "deep" literally refers to the number of layers. Modern networks for tasks like language understanding can have **billions of parameters**.

An interesting parallel from the lecture: humans and machines both separate *learning* from *doing*. Our brains learn best in a state of **relaxed alertness** ("rest and digest"), while stress ("fight-or-flight") is for execution. Neural networks mirror this with distinct **training** and **prediction (inference)** phases — a model is either learning from data or applying what it learned, not both at once.

---

### 2. A Brief History of AI

#### 2.1 Early Neural Networks (1950s)

Almost as soon as computers existed, researchers built **artificial neural networks inspired by the biology of the brain**. They had modest successes (reading handwritten digits, playing backgammon), but they were held back by one huge problem: **not enough compute**. Training a network takes enormous numbers of calculations — imagine trying to run today's training loops on 1950s hardware.

As a result, neural networks were outclassed by conventional programs running on the **Von Neumann architecture** (the design behind modern CPUs: a processor executing explicit instructions stored in memory).

#### 2.2 Expert Systems

For decades, the dominant approach to "intelligent" software was the **expert system**: a huge collection of hand-written rules meant to mimic a human subject-matter expert.

**Examples:** a stock-trading bot that trades based on user-defined limits; a medical diagnosis tool driven by a programmed symptom checklist.

**The fatal limitation:** every rule had to be *understood, defined, and programmed by a human*. The system could never do more than its engineers could articulate in code.

> **The three-images thought experiment:** show a human photos of a cat, a car, and an ocean wave — they classify them instantly. Now try to *write the rules* a computer could follow to do the same. What makes a cat a cat, in terms of pixels? You can't write it down. Tasks that are trivial for humans can be nearly impossible to express as explicit rules. This is exactly where expert systems break down.

#### 2.3 The Insight: How Do Children Learn?

Children aren't given rules for recognizing cats. Instead we:

1. **Expose them to lots of data** (they see many cats),
2. **Give them the correct answer** ("yes, that's a cat" / "no, that's a dog"),
3. Let them **pick up the important patterns on their own** through trial and error.

Deep learning works the same way. This analogy is the conceptual core of the whole course: *data + feedback + iteration → learned patterns*.

---

### 3. The Deep Learning Revolution

Through the 1990s and early 2000s, powerful AI looked like a pipe dream. Two ingredients arrived together and changed everything:

#### 3.1 Data 📊

A network can only learn what a cat is by seeing **many** cats — and many *non-cats*. The internet era produced exactly the mountains of labeled data (images, text, clicks) that networks need.

#### 3.2 Computing Power — and Why GPUs 🖥️

Under the hood, **neural networks are matrix multiplication machines**. What else is built on massive matrix math? **Computer graphics** — 3D scenes are millions of tiny triangles being transformed and rotated via matrices.

Because both problems share the same mathematical foundation, the hardware built for one (the **GPU**) turned out to be perfect for the other:

| | CPU | GPU |
|---|---|---|
| Cores | ~4–16 powerful cores | Thousands of simpler cores |
| Designed for | Sequential, complex logic | Massive parallel arithmetic |
| DL training | Possible but painfully slow | Ideal — millions of simple calculations at once |

The individual calculations in training aren't complex — there are just *astronomically many* of them, and they can run in parallel. That's a GPU's home turf.

---

### 4. Deep Learning Flips Programming on Its Head

This is the single most important conceptual shift in the course.

**Traditional programming (building a classifier):**

```
Rules (written by humans) + Data  →  Answers
```

You articulate the rules, program them, and the program applies them to data.

**Machine learning:**

```
Data + Answers (labels)  →  Rules (learned by the model)
```

You collect input/output examples (the **dataset**), and the model repeatedly *guesses*, gets told whether it was right, and adjusts. Over the course of training it discovers the rules on its own — including rules no human could have written down.

#### When to choose which?

| Situation | Approach |
|---|---|
| Rules are clear and can be stated precisely (e.g., route customers to sales reps by region) | **Classic programming** — simpler, cheaper, explainable |
| Rules are nuanced/complex and humans can't articulate them (e.g., "is this a cat?") | **Deep learning** |

Deep learning is *not* always the right tool. If you can write the rules, write the rules.

---

### 5. Where Deep Learning Is Transforming the World

- **Computer Vision** — teaching machines to see and interpret images the way humans do. The field predates DL, but the flood of tagged, high-quality images made DL dominant here.
- **Natural Language Processing** — real-time translation, voice recognition, virtual assistants.
- **Recommender Systems** — curated feeds (Facebook), music (Spotify), video (YouTube/Netflix), shopping and ads (Amazon).
- **Reinforcement Learning** — agents that learn by acting and receiving rewards; famously AlphaGo beating the world Go champion, and bots competing with pros at StarCraft and DOTA.

---

### 6. Tooling for This Course

The three major deep learning frameworks:

| Framework | Backed by |
|---|---|
| TensorFlow + Keras | Google |
| **PyTorch** ← used in this course | Meta |
| MXNet | Apache |

None is a clear "winner" — this course uses **PyTorch**, and the labs run on a GPU server accessed through **JupyterLab** notebooks. It's worth gaining exposure to the others over time.

---
---

## 🧪 Section B — Lab: Image Classification with MNIST (`01_mnist.ipynb`)

The "Hello World" of deep learning: train a model to classify handwritten digits.

### 1. The Problem and Why DL Solves It

**Image classification** = given an image the program has *never seen*, assign it to the correct class. With traditional programming this is near impossible — nobody can write pixel-level rules for "this is a 7." Deep learning instead does **pattern recognition by trial and error**: show the network enough labeled data, give it feedback on its guesses, and through massive iteration it finds its own internal conditions for classifying correctly.

### 2. The MNIST Dataset

- **70,000 grayscale images** of handwritten digits (0–9), each **28 × 28 pixels**.
- Historically a major benchmark; today it's trivial, but it survives as a **debugging tool** — if a fancy new architecture can't learn MNIST, it probably can't learn anything harder.

#### 2.1 Train vs. Validation — the flashcard analogy 🃏

We need images (**`X`**) and their correct labels (**`Y`**), split into two groups:

| Split | Contents | Purpose |
|---|---|---|
| `x_train`, `y_train` | 60,000 images + labels | The deck the model **studies with** |
| `x_valid`, `y_valid` | 10,000 images + labels | A **separate deck the teacher quizzes with** |

The point of the separate validation set: a student who memorized one deck of flashcards hasn't necessarily *learned the concept*. Quizzing with unseen cards reveals whether real understanding (generalization) happened. Same for models.

```python
import torchvision

train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)
```

> Note: torchvision calls the second split "Test", but this course uses it for validation. (Strictly: *validation* data guides development; *test* data is held back for a final, one-time evaluation.)

Each item in the dataset is an `(image, label)` pair — the image is a **PIL Image**, the label a plain `int`:

```python
x_0, y_0 = train_set[0]   # x_0: PIL image of a digit, y_0: 5
```

### 3. Tensors

**Definition by escalation:** a *vector* is a 1-D array, a *matrix* is 2-D, and a **tensor** is the general n-dimensional array. Modern DL frameworks are essentially high-performance tensor processors.

**Concrete example of a 3-D tensor:** your screen — dimensions of width × height × color channel. This is also *why GPUs excel at tensors*: games manipulate pixels with the same matrix math networks use.

#### 3.1 Converting an image to a tensor

```python
import torchvision.transforms.v2 as transforms

trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)
```

Three things `ToTensor` gives you, worth memorizing:

1. **dtype** becomes `torch.float32`.
2. **Value range rescales** from PIL's integer `[0, 255]` to float `[0.0, 1.0]` — a form of normalization that helps training.
3. **Shape follows the `C × H × W` convention** (channel, height, width). MNIST is grayscale, so: `torch.Size([1, 28, 28])`.

#### 3.2 CPU vs GPU tensors

Tensors live on the CPU by default. `.cuda()` moves one to the GPU but **crashes if no GPU exists**. The robust pattern — set a `device` once, then `.to(device)` everywhere:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_0_tensor.to(device)   # fast on GPU, still works on CPU
```

This one idiom makes your code portable between your GPU workstation and a laptop.

### 4. Preparing Data for Training

#### 4.1 Transforms

Attach the transform pipeline to the datasets so every image is converted on the fly:

```python
train_set.transform = trans
valid_set.transform = trans
```

#### 4.2 DataLoaders and batching

If the dataset is the deck of flashcards, the **DataLoader** defines *how you draw cards from the deck*:

```python
from torch.utils.data import DataLoader

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
```

Why not show the model the whole dataset at once?

- It would blow up memory/compute, **and**
- research shows smaller batches actually train *more efficiently*.

Details to remember:

- **`batch_size` is a developer's choice** (a *hyperparameter*). 32 or 64 works for many problems and is a common default.
- **Shuffle the training data** so the model doesn't learn the order of the deck. **No shuffling needed for validation** — the model isn't learning there; we still batch it just to avoid memory errors.

### 5. Building the Model

The network is a stack of **layers**, each performing a mathematical operation on its input before passing the result on. Our "Hello World" architecture has four components: **Flatten → Input layer → Hidden layer → Output layer**.

#### 5.1 Flatten — from image to vector

`nn.Flatten` squashes an n-dimensional image into a single 1-D vector so it can feed a dense layer.

⚠️ **Gotcha worth remembering:** Flatten expects a **batch** of data. Give it a bare 2-D matrix and it treats the rows as separate items — apparently doing nothing:

```python
test_matrix = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])

nn.Flatten()(test_matrix)          # "nothing happens" — no batch dim!
nn.Flatten()(test_matrix[None, :]) # → tensor([[1,2,3,4,5,6,7,8,9]]) ✓
```

`[None, :]` adds a new leading dimension (the batch dimension). And **order matters** — `[:, None]` puts the new axis in the wrong place and produces a different (wrong) result.

#### 5.2 Input layer

A `nn.Linear` layer that is **densely connected** — every neuron (and its weights) affects every neuron in the next layer. It needs to know two sizes:

```python
input_size = 1 * 28 * 28   # C × H × W of a flattened MNIST image = 784
nn.Linear(input_size, 512)
```

Why **512 neurons**? Honestly: it's a judgment call — "the science in data science" — about matching the statistical complexity of the dataset. Experiment with it and build intuition.

Each linear layer is followed by **ReLU**, an *activation function*. For now the key idea: without it the network could only ever learn straight-line (linear) relationships; ReLU lets it make more sophisticated, non-linear guesses. (Covered properly in a later part.)

#### 5.3 Hidden layer

Another dense layer sandwiched between input and output. Its input size must equal the previous layer's neuron count (each neuron outputs one number):

```python
nn.Linear(512, 512)
```

#### 5.4 Output layer

Ten classes (digits 0–9) → **10 output neurons, one per class**. The bigger a neuron's value relative to the others, the more the model believes the image belongs to that class. No ReLU here — the raw scores go to the loss function instead.

#### 5.5 Assembling and compiling

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),
    nn.Linear(512, 512),         # Hidden
    nn.ReLU(),
    nn.Linear(512, n_classes)    # Output (n_classes = 10)
)
model.to(device)                 # move to GPU
model = torch.compile(model)     # PyTorch 2: compile for speed
```

(In the notebook the layers are built up in a Python list and unpacked with `nn.Sequential(*layers)` — the `*` operator turns the list into a sequence of arguments, since `Sequential` doesn't accept a list directly.)

To check which device a model lives on, check its parameters: `next(model.parameters()).device`.

### 6. Training the Model

"Training a model" is also called **fitting** — the shape of the model literally changes over time to fit the data.

#### 6.1 Loss function and optimizer — the teacher and the tutor

Two objects drive learning:

```python
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
```

- **Loss function** = the *teacher grading answers*. `CrossEntropyLoss` is designed for grading "pick the correct category out of a group" predictions.
- **Optimizer** = the *tutor* that tells the model **how to learn from that grade** to do better next time. `Adam` is a solid general-purpose choice.

#### 6.2 Accuracy — a human-readable metric

Loss values drive learning but are hard for humans to interpret, so we also track **accuracy** = correct predictions ÷ total predictions. Since data arrives in batches, each batch contributes its fraction and they sum to the total:

```python
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)      # index of the largest score = predicted class
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N
```

#### 6.3 The train function — everything comes together

```python
def train():
    loss, accuracy = 0, 0
    model.train()                              # training mode
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)      # move batch to GPU
        output = model(x)                      # 1. forward pass (guess)
        optimizer.zero_grad()                  # 2. clear old gradients
        batch_loss = loss_function(output, y)  # 3. grade the guess
        batch_loss.backward()                  # 4. backpropagate (how was I wrong?)
        optimizer.step()                       # 5. update weights (learn)

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
```

The five numbered steps are **the** training loop pattern — you'll write this shape for the rest of your PyTorch life: *forward → zero grad → loss → backward → step*.

#### 6.4 The validate function — spot the differences

```python
def validate():
    loss, accuracy = 0, 0
    model.eval()                               # ① evaluation mode
    with torch.no_grad():                      # ② no gradient tracking
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
```

Three deliberate differences from `train()` — this is a classic exam/interview question:

1. **`model.eval()`** instead of `model.train()` — switches layer behavior to inference mode.
2. **`torch.no_grad()`** — no gradients are computed, saving memory and time, because…
3. **No `zero_grad` / `backward` / `step`** — the model is being *quizzed*, not *taught*. No learning happens on validation data, ever.

#### 6.5 Epochs — going through the deck multiple times

An **epoch** = one complete pass through the entire dataset. Just as a student needs several passes through the flashcards, the model trains for multiple epochs, validating after each to track progress:

```python
epochs = 5
for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train()
    validate()
```

On MNIST, both training and validation accuracy climb close to **100%** within a few epochs.

### 7. Making a Prediction (Inference)

A trained model can be called like a function. It outputs 10 numbers — one per output neuron, where index = digit (index 0 is the score for "0", etc.). `argmax` picks the winner:

```python
prediction = model(x_0_gpu)
prediction.argmax(dim=1, keepdim=True)   # → tensor([[5]]) — matches y_0 ✓
```

Using a trained model on new, unseen data is called **inference** — explored properly in a later notebook.

### 8. Housekeeping: Clear GPU Memory

Before moving to the next notebook, shut down the kernel to release GPU memory:

```python
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

---

## 🔑 Key Takeaways

1. **DL flips programming:** instead of *rules + data → answers*, you provide *data + answers* and the model learns the *rules* — like a child learning by exposure and correction.
2. **The revolution = data + GPUs.** Networks are matrix-multiplication machines; GPUs (thousands of parallel cores) were built for exactly that math.
3. **DL is not always the answer** — if you can articulate the rules, just program them.
4. **Train/validation split is non-negotiable:** study with one flashcard deck, quiz with another; that's how you detect real learning vs memorization.
5. **`ToTensor`** → float32, values rescaled to `[0, 1]`, shape `C × H × W`. Use `.to(device)` for GPU-portable code.
6. **DataLoaders batch and shuffle** — batch size is a hyperparameter (32/64 are common); shuffle training data only.
7. **Architecture pattern:** `Flatten → Linear + ReLU → Linear + ReLU → Linear(n_classes)`; output neurons are per-class scores, `argmax` picks the prediction.
8. **Loss = the grade, optimizer = how to improve.** The training loop is always *forward → zero_grad → loss → backward → step*.
9. **Validation differs by three things:** `model.eval()`, `torch.no_grad()`, and no weight updates.
10. **An epoch** is one full pass through the data; MNIST reaches ~100% accuracy in ~5 epochs — and remains a great sanity-check benchmark for any new architecture.
