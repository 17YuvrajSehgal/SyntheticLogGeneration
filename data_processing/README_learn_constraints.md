# Constraint Learning Module

The `learn_constraints.py` script is the core "Knowledge Base" builder for the Synthetic Kernel Log generation system. It analyzes real traces to learn the fundamental rules of the Linux kernel, ensuring that synthetic traces respect physical and logical invariants.

## Usage
```bash
python data_processing/learn_constraints.py \
    --real_glob "dataset/window_shards/**/*.npz" \
    --output dataset/constraints_universal.json
```

## Learned Invariants

The script iterates through every event in the dataset and learns the following constraints:

### 1. Transition Graph (Logic)
*   **What**: Legal transitions between events $(E_t \to E_{t+1})$.
*   **Why**: To prevent nonsensical sequences (e.g., `file_close` before `file_open`).
*   **Storage**: Adjacency List.

### 2. Temporal Bounds (Physics)
*   **What**: Minimum and Maximum allowed time delta (`dt`) for each event type.
*   **Why**: Kernel operations have physical speed limits. A `context_switch` cannot happen instantly; a disk I/O has latency.
*   **Storage**: `{min, max, allowed_set}` per Event ID.

### 3. CPU Affinity (Topology)
*   **What**: Which CPU cores trigger specific events.
*   **Why**: Some interrupts or per-cpu workers are pinned to specific cores.
*   **Storage**: Set of allowed CPUs per Event ID.

### 4. Thread Identity (TID)
*   **What**: Which Thread ID buckets trigger specific events.
*   **Why**: Detecting if an event is exclusive to specific threads (e.g., PID 0/Swapper).
*   **Storage**: Set of allowed TID Buckets per Event ID.

### 5. Semantic Context (Comm, FD, Ret)
*   **Comm**: Command Name IDs allowed for each event. (e.g., does `postgres` trigger this syscall?)
*   **FD**: File Descriptor IDs allowed. (e.g., is this event strictly for STDIN/STDOUT?)
*   **Ret**: Return Values allowed. (e.g., does this syscall ever return -1?)

## Output Format
The resulting `constraints.json` is used by:
1.  **`validate.py`**: To score synthetic traces.
2.  **`repair.py`**: To fix invalid generated events by sampling from the learned "Allowed Sets."
