# Dataset Structure and Semantics

This dataset contains a **windowed representation of kernel execution traces (LTTng)** designed for **sequence modeling** tasks such as generative modeling, anomaly detection, and representation learning.

---

## 1. High-level overview

A kernel trace is originally a **single, very long sequence of events** describing system behavior over time.
To make this data usable for machine learning models, the trace is split into **fixed-length windows**.

Each window represents a **short sequence of consecutive kernel events** with associated timing and CPU information.

---

## 2. Files and format

Each dataset file is stored as a NumPy `.npz` archive containing three arrays:

```text
event: (N, L) int32
dt   : (N, L) uint8
cpu  : (N, L) uint8
```

Where:

* **N** = number of windows (sequences)
* **L** = window length (number of events per window)

Example:

```text
event: shape = (100000, 200)
```

This means:

* The dataset contains **100,000 independent sequences**
* Each sequence is **200 events long**

---

## 3. What is a window?

A **window** is a fixed-length **subsequence** extracted from the original trace.

Conceptually:

```text
Raw trace (very long):
e₁ → e₂ → e₃ → ... → eₘ

Windowing (L = 200):
Window 0: e₁   → e₂   → ... → e₂₀₀
Window 1: e₂₀₁ → e₂₀₂ → ... → e₄₀₀
Window 2: e₄₀₁ → e₄₀₂ → ... → e₆₀₀
...
```

Each window is treated as a **separate training example**.

---

## 4. What is a sequence?

In this dataset, a **sequence** refers to one window.

Each sequence consists of **L = 200 time steps**, and at each time step the following information is recorded:

| Field      | Description                                                |
| ---------- | ---------------------------------------------------------- |
| `event[t]` | Encoded kernel event ID                                    |
| `dt[t]`    | Time delta since the previous event (quantized / bucketed) |
| `cpu[t]`   | CPU core on which the event occurred                       |

So a sequence looks like:

```text
(event₁, dt₁, cpu₁),
(event₂, dt₂, cpu₂),
...
(event₂₀₀, dt₂₀₀, cpu₂₀₀)
```

---

## 5. Meaning of each array

### `event` (int32)

* Categorical ID representing a kernel event
* Mapped to human-readable names via `vocab.json`
* Common events appear frequently (e.g., scheduler or syscall events)
* Rare events capture interrupts, exceptions, or uncommon paths

### `dt` (uint8)

* Time difference between consecutive events
* Stored as **bucketed / quantized values**
* Captures temporal dynamics while keeping memory usage small
* Smaller values indicate bursts of activity; larger values indicate idle gaps

### `cpu` (uint8)

* CPU core ID on which the event occurred
* Enables learning of multi-core execution patterns
* Reveals CPU affinity and migration behavior

---

## 6. Interpretation of dimensions

Using `event.shape = (100000, 200)` as an example:

| Dimension | Meaning                                     |
| --------- | ------------------------------------------- |
| 100000    | Number of windows (sequences)               |
| 200       | Length of each sequence (events per window) |

**Important:**

* The model sees each row independently.
* The dataset does **not** represent one giant sequence of length 20 million.
* Long-term behavior is captured statistically across many windows.

---

## 7. Why windowing is used

Windowing is essential because:

* Machine learning models require **fixed-length inputs**
* Very long traces cannot fit in memory
* Training benefits from many independent samples
* Windows allow efficient batching on GPUs

This design trades **global continuity** for **local temporal structure**, which is sufficient for learning realistic kernel behavior.

---

## 8. Dataset characteristics

Typical properties of the data:

* Strong repetition in steady-state execution
* Skewed event distributions (few events dominate)
* Stable timing patterns with occasional bursts
* CPU imbalance (e.g., CPU 0 often dominates)
* Rich cross-field dependencies between event type, timing, and CPU

These properties reflect **realistic system behavior**.

---

## 9. How models use this data

Models are trained to learn:

* Temporal ordering of events
* Timing distributions conditioned on event type
* CPU affinity and switching behavior
* Cross-field semantics across `event`, `dt`, and `cpu`

Each window is treated as a **self-contained behavior snapshot**.

---

## 10. Key takeaway

> **Each dataset file contains many short sequences (windows).
> Each window is a fixed-length sequence of kernel events with timing and CPU context.
> Together, these windows statistically represent the behavior of the full trace.**