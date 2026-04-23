# Day 4: Attention Mechanism Theory - Methods Overview

## Introduction

This document covers the **Attention mechanism**, the foundation of modern language models (LLMs). Traditional architectures like RNN/LSTM struggle to learn long-range dependencies. The attention mechanism solves this by dynamically computing **how much "attention"** each word in a sentence should pay to every other word.

---

## 1. The Problem: RNN/LSTM Forgetting Issue

### Why RNN Is Insufficient

When processing a sentence, a **hidden state** vector carries information from previous words:

```
"When I was young, the yellow cat I loved... do you remember it?"
 t=1   t=2  t=3   t=4  t=5   t=6  t=7   t=8    t=9  t=10   t=11
```

Problem: At `t=11`, when processing `"it"`, the information about `"cat"` at `t=6` may have **gradually faded** in the hidden state.

### Numerical Perspective: Information Loss

In an RNN, the hidden state is updated as:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t)
```

The `tanh` activation at every step causes gradient shrinkage:
- After 10 steps: ~0.4^10 ≈ **0.0001** (vanishing gradient)
- After 50 steps: effectively zero

### LSTM Partially Solves This, But Not Entirely

LSTM adds a memory cell with **gate** mechanisms:
- Forget gate: selects what information to erase
- Input gate: determines what new info to add
- Output gate: controls what to share

However, LSTM still suffers from:
- No parallel computation (forced sequential processing)
- Information loss on very long dependencies
- High computational cost

### Solution: Attention Mechanism

> Every word can **directly access** every other word — distance doesn't matter!

```
"The animal couldn't cross the street because it was too wide."
                                              ↕   ↕
                                  "wide" → "street" relationship!
                                  (high attention score)
```

---

## 2. Attention Mechanism: The Intuition

### The Database Analogy

The attention mechanism works like a **fuzzy (soft) database lookup**:

| Concept | Database | Attention Mechanism |
|---------|----------|---------------------|
| What am I looking for? | Query | **Q** — "What is this word searching for?" |
| What is it compared against? | Key | **K** — "How does this word identify itself?" |
| What is returned? | Value | **V** — "What is the actual content of this word?" |

### Intuitive Example

```
Query: "What does the cat eat?" (Q)

Key-Value pairs:
  K1="cat"    V1="furry animal"
  K2="eats"   V2="action of consuming food"
  K3="milk"   V3="white liquid"
  K4="pen"    V4="writing instrument"

Similarities:
  Q·K1 = 0.9  (very relevant!)
  Q·K2 = 0.7  (relevant)
  Q·K3 = 0.6  (somewhat relevant)
  Q·K4 = 0.1  (irrelevant)

Result = 0.9 × V1 + 0.7 × V2 + 0.6 × V3 + 0.1 × V4
       = weighted blend of information
```

---

## 3. Query, Key, Value Matrices

### Why Three Separate Matrices?

Each word embedding is **projected into 3 different roles**:

```python
# d_model: embedding dimension (e.g. 512)
# d_k: key/query dimension (e.g. 64)
# d_v: value dimension (e.g. 64)

W_Q = nn.Linear(d_model, d_k)  # Query matrix
W_K = nn.Linear(d_model, d_k)  # Key matrix
W_V = nn.Linear(d_model, d_v)  # Value matrix

# x: [seq_len, d_model] input
Q = x @ W_Q  # [seq_len, d_k] — "What am I looking for?"
K = x @ W_K  # [seq_len, d_k] — "What do I offer?"
V = x @ W_V  # [seq_len, d_v] — "What is my actual content?"
```

### The Role of Each Matrix

**Query (Q):**
- "What context does my word need attention for?"
- The question this word is asking of other words
- Example: "I (`cat`) want information about what?"

**Key (K):**
- "How can you find me? What's my label?"
- The vector each word uses to announce "this is what I am"
- Example: "I (`milk`) am the right word for an object."

**Value (V):**
- "If you attend to me, here's what I give you."
- The actual information content — multiplied by attention weights
- Example: "Content: white, liquid, nutritious, cats love it..."

### Why Are Q and K Separate?

If Q and K were identical, each word would maximally attend to itself. Separate matrices:
- Allow asymmetric relationships
- Provide learning flexibility
- Separate "asker" from "answerer" roles

---

## 4. Scaled Dot-Product Attention

### Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Step-by-Step Computation

**Step 1: Compute raw scores**
```python
scores = Q @ K.T  # [seq_len, seq_len]
# Each row: one word's raw attention scores toward all words
```

**Step 2: Scale (divide by √d_k)**
```python
scores = scores / math.sqrt(d_k)
# Why? Large d_k causes very large dot products
# → Softmax produces extremely peaked distributions → gradients vanish
```

**Step 3: Apply softmax**
```python
weights = softmax(scores, dim=-1)  # [seq_len, seq_len]
# Each row sums to 1, values in [0,1]
# = attention weights
```

**Step 4: Weighted sum of values**
```python
output = weights @ V  # [seq_len, d_v]
# For each token: a weighted blend of all value vectors
```

### Why Scale by √d_k?

For random vectors of dimension d_k:
- Variance of dot product: d_k
- Standard deviation: √d_k

Dividing by √d_k normalizes variance to 1 → softmax works well.

```
d_k=64:  divide by √64 = 8
d_k=512: divide by √512 ≈ 22.6
```

---

## 5. Attention Weights: "Understanding" a Sentence

### Example: Coreference Resolution

```
"The animal couldn't cross the street because it was too wide."
```

When processing `"wide"`, attention weights might be:

```
wide → The:     0.03
wide → animal:  0.07
wide → street:  0.62  ← highest attention!
wide → because: 0.05
wide → it:      0.16
wide → was:     0.07
```

The model has learned that `"wide"` is associated with `"street"`!

### Attention Matrix Visualization

```
           The  animal  street  because  it  was  wide
The      [ 0.80   0.05    0.04    0.02  0.03 0.02  0.04 ]
animal   [ 0.03   0.75    0.09    0.04  0.03 0.02  0.04 ]
...
wide     [ 0.03   0.07    0.62    0.05  0.16 0.07   --  ]
```

Each row is a word's attention distribution (sums to 1).

---

## 6. Multi-Head Attention

### Why Multiple Heads?

A single attention mechanism can only learn **one type of relationship**. Real language has many:

- Syntactic: Subject-verb relationships
- Semantic: Meaning similarity
- Coreference: Pronouns and their antecedents
- Positional: Adjacent word relationships

### Structure

```python
# h parallel attention mechanisms
class MultiHeadAttention:
    def __init__(self, d_model, h):
        self.h = h                    # Number of heads (e.g. 8)
        self.d_k = d_model // h       # Dimension per head (e.g. 512//8=64)
        
        # Separate Q, K, V matrices for each head
        self.W_Q = [Linear(d_model, d_k) for _ in range(h)]
        self.W_K = [Linear(d_model, d_k) for _ in range(h)]
        self.W_V = [Linear(d_model, d_k) for _ in range(h)]
        
        # Matrix to combine outputs
        self.W_O = Linear(h * d_k, d_model)
    
    def forward(self, x):
        heads = []
        for i in range(self.h):
            Q_i = x @ self.W_Q[i]
            K_i = x @ self.W_K[i]
            V_i = x @ self.W_V[i]
            head_i = scaled_dot_product_attention(Q_i, K_i, V_i)
            heads.append(head_i)
        
        # Concatenate all head outputs
        concatenated = concat(heads, dim=-1)  # [seq_len, h*d_k]
        output = concatenated @ self.W_O       # [seq_len, d_model]
        return output
```

### GPT-2 / BERT Parameters

| Model | d_model | Heads (h) | d_k (per head) |
|-------|---------|-----------|----------------|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Large | 1280 | 20 | 64 |
| BERT-Base | 768 | 12 | 64 |
| GPT-4 (estimated) | 12800 | 96 | ~128 |

---

## 7. Masked Attention

### Why Masking?

Language models enforce **causality**: the model cannot see future tokens.

```
"The cat sits on the mat"

t=1: "The"  → can only see "The"
t=2: "cat"  → can see "The cat"
t=3: "sits" → can see "The cat sits"
t=4: "on"   → can see "The cat sits on"
```

### Implementation: Negative Infinity Mask

```python
import torch

def create_causal_mask(seq_len):
    # Lower triangular matrix: 1 = attend, 0 = don't attend
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# Applied to scores:
# positions where mask=0 get -inf → after softmax they become 0
masked_scores = scores.masked_fill(mask == 0, float('-inf'))
weights = softmax(masked_scores, dim=-1)
```

### Visual

```
Positions that can be attended to (✓ = yes, ✗ = no):

          The  cat  sits  on  mat
The     [  ✓    ✗    ✗    ✗   ✗  ]
cat     [  ✓    ✓    ✗    ✗   ✗  ]
sits    [  ✓    ✓    ✓    ✗   ✗  ]
on      [  ✓    ✓    ✓    ✓   ✗  ]
mat     [  ✓    ✓    ✓    ✓   ✓  ]
```

---

## 8. Positional Encoding

### The Problem

Attention is **position-agnostic** — it doesn't care about order:

```
"The cat chased the mouse" ≡ "The mouse chased the cat" (from attention's view)
```

### Sinusoidal Encoding (Original Transformer)

A fixed vector is added to each position:

```python
def positional_encoding(seq_len, d_model):
    PE = zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]   = sin(pos / 10000^(2i/d_model))
            PE[pos, i+1] = cos(pos / 10000^(2i/d_model))
    return PE

# Usage:
x = embedding(tokens) + positional_encoding(seq_len, d_model)
```

### Why Sinusoidal?

- Fixed patterns: Each position receives a unique vector
- Relative position: `PE(pos+k)` can be computed via linear transform from `PE(pos)`
- Arbitrary length: Theoretically generalizes beyond training length

### Modern: Learnable Positional Embeddings

GPT-2, BERT learn a separate embedding for each position:
```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
x = token_embedding(tokens) + self.pos_embedding(positions)
```

---

## 9. Full Transformer Block Structure

```
Input: x [batch, seq_len, d_model]
         │
         ▼
┌─────────────────────────────┐
│  Multi-Head Attention        │
│  MultiHead(Q, K, V)          │
└────────────┬────────────────┘
             │
         [Residual] ──────────────── x
             │                        │
             └────────────────────────┘
             ▼
         LayerNorm(x + Attention_output)
             │
             ▼
┌─────────────────────────────┐
│  Feed-Forward Network (FFN) │
│  Linear → ReLU → Linear     │
└────────────┬────────────────┘
             │
         [Residual] ──────────────── x
             │                        │
             └────────────────────────┘
             ▼
         LayerNorm(x + FFN_output)
             │
             ▼
         Output: [batch, seq_len, d_model]
```

### Residual Connection

Why important?
- Enables gradient flow through deep networks
- Implements "learn the change, not the full output" principle
- Makes stacks of many layers trainable

### Layer Normalization

```python
# Instead of batch normalization:
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β
# Normalized per token
# Stabilizes training
```

---

## 10. Computational Complexity

### Attention: The O(n²) Issue

```
For sequence length n, attention matrix is: n × n
→ n=512:    262,144 operations
→ n=2048:   4,194,304 operations (16× more!)
→ n=128000: ~16 billion operations (GPT-4 context)
```

This is why processing long contexts is **expensive**.

### Modern Solutions

| Method | Complexity | Approach |
|--------|------------|----------|
| Standard Attention | O(n²) | Baseline |
| Sparse Attention | O(n·√n) | Attend to specific positions |
| Linear Attention | O(n) | Kernel method |
| Flash Attention | O(n²) but fast | IO-aware CUDA implementation |
| Sliding Window | O(n·w) | Attend within a window |

---

## 11. Self-Attention vs Cross-Attention

### Self-Attention

Q, K, V all come from the **same source**:
```python
# Inside encoder or decoder
Q = K = V = x  # Same sequence
# Usage: BERT, GPT encoder layers
```

### Cross-Attention

Q from one source, K and V from another:
```python
# Encoder-Decoder architecture
Q = decoder_state      # "What am I trying to produce?"
K = V = encoder_output # "What's in the source sentence?"
# Usage: Translation, summarization
```

### Practical Example: Translation

```
Source: "The cat sat" → Encoder → encoder_output
Target: "Kedi oturdu" → Decoder

When producing "Kedi":
  Q = state of "Kedi"
  K, V = encoder outputs for ["The", "cat", "sat"]
  → High attention on "cat"!
```

---

## 12. Real-World Applications

### BERT: Encoder-Only (Bidirectional)

- All tokens can attend both left and right
- No causal mask (masked language modeling training is different)
- Usage: Classification, NER, QA

### GPT: Decoder-Only (Unidirectional)

- Only attends to previous tokens (causal mask)
- Autoregressive text generation
- Usage: Text generation, completion

### T5/BART: Encoder-Decoder (Full)

- Encoder: bidirectional self-attention
- Decoder: causal self-attention + cross-attention
- Usage: Translation, summarization, QA

---

## 13. Key Takeaways

1. **RNN forgets**: Long-range dependencies are lost due to vanishing gradients
2. **Attention provides direct access**: Every word can reach every other word
3. **Q-K-V trio**: Query, Key, Value — "what I seek, what I offer, what I give"
4. **√d_k scaling**: Critical for numerical stability
5. **Multi-head attention**: Learns different relationship types in parallel
6. **Causal mask**: Hides future tokens — essential for autoregressive generation
7. **Positional encoding**: The way to inject order information
8. **O(n²) complexity**: Bottleneck for long contexts

---

## 14. What's Next: Transformer Architecture

When attention mechanisms are stacked together they form the **Transformer** — the architecture at the core of GPT, BERT, and every modern LLM.

```
"Attention is All You Need" — Vaswani et al., 2017
```

Next: Exploring the full Transformer architecture with Encoder, Decoder, and training.

---

## Additional Resources

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer (Vaswani et al., 2017)
- [BERT](https://arxiv.org/abs/1810.04805) — Encoder-only Transformer
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Fast and memory-efficient attention

### Interactive Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide
- [Attention Visualizer](https://huggingface.co/spaces/exbert-project/exbert) — View real BERT attention heads
- [BertViz](https://github.com/jessevig/bertviz) — Visualize Transformer attention

### Video Resources
- [Andrej Karpathy - Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — GPT from scratch
- [3Blue1Brown - Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc) — Visual explanation
