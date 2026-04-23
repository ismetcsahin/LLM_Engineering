# Day 3: Word Embeddings and Multi-Dimensional Space - Methods Overview

## Introduction

This document covers how words are converted into dense numerical vectors that capture semantic properties. The transition from token IDs (integers) to vectors that encode semantic relationships is fundamental to modern NLP. Vector arithmetic like "King - Man + Woman = Queen" demonstrates how embeddings capture meaning.

---

## 1. What Is an Embedding?

### Definition
An embedding is a fixed-size dense vector representing a word or token. Typically 100-300 dimensions, where each dimension encodes a semantic feature of the word.

### Why Do We Need Embeddings?

Token IDs (integers) don't capture meaning:
- Token ID 5234 ("cat") and 8912 ("dog") are just numbers
- No relationship between similar words
- Mathematical operations are meaningless

### One-Hot Encoding: The Naive Approach

```python
Vocabulary: ['cat', 'dog', 'king', 'queen']
cat:    [1, 0, 0, 0]
dog:    [0, 1, 0, 0]
king:   [0, 0, 1, 0]
queen:  [0, 0, 0, 1]
```

Problems:
- Sparse: Most values are zero
- High-dimensional: Vocab size = vector size
- Meaningless: All words equally distant
- No semantic information: cat-dog distance = cat-king distance

### Dense Embeddings: The Solution

```python
cat:    [0.8, 0.9, 0.1, ...]  # 100-300 dims
dog:    [0.7, 0.85, 0.15, ...]
king:   [0.1, 0.2, 0.9, ...]
queen:  [0.15, 0.25, 0.85, ...]
```

Advantages:
- Dense: All values are meaningful
- Low-dimensional: Fixed size (independent of vocab)
- Semantic: Similar words have similar vectors
- Learnable: Automatically learned from data

---

## 2. The Distributional Hypothesis

### Core Principle

> "You shall know a word by the company it keeps" — J.R. Firth (1957)

Words appearing in similar contexts have similar meanings.

### Examples

```
"The cat drinks milk"
"The dog drinks milk"
→ cat and dog appear in similar contexts → similar meanings

"The king wears a crown"
"The queen wears a crown"
→ king and queen appear in similar contexts → similar meanings
```

### Why It Works

- Context defines meaning
- Statistical co-occurrence reveals semantic similarity
- No supervision required — learns from raw text

---

## 3. Word2Vec

### What It Is

An efficient neural network architecture for learning word embeddings, introduced by Google in 2013.

### Two Architectures

#### 1. Skip-gram (More Common)
- Task: Given a center word, predict context words
- Example: "the cat sits on the mat"
  - Center: "sits" → Context: ["cat", "on"] (window=1)
  - Center: "cat" → Context: ["sits"] (window=1)

#### 2. CBOW (Continuous Bag of Words)
- Task: Given context words, predict the center word
- Example: Context: ["cat", "on"] → Center: "sits"

### Skip-gram Architecture

```
Input: Center word (one-hot)
  ↓
Embedding Layer (W_in): V × D matrix
  ↓
Hidden Layer: D-dimensional vector (the embedding!)
  ↓
Output Layer (W_out): D × V matrix
  ↓
Softmax: Probability distribution over context words
```

- V = vocabulary size
- D = embedding dimension (typical: 100-300)

### Training Process

1. Generate (center, context) pairs from corpus
2. Initialize random embeddings
3. For each pair:
   - Forward pass: Get center embedding, predict context
   - Compute loss: Cross-entropy
   - Backpropagation: Update embeddings
4. Repeat over millions of words

### Negative Sampling

Computing softmax is expensive (over entire vocabulary). Solution:
- 1 real context word (positive sample)
- K random words (negative samples, typical K=5-20)
- Binary classification: real context or noise?

This speeds up training by 100×.

---

## 4. Cosine Similarity

### Definition

The cosine of the angle between two vectors — measures semantic similarity.

```python
cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)
```

- Range: [-1, 1]
- 1 = same direction (very similar)
- 0 = orthogonal (unrelated)
- -1 = opposite direction (opposite meanings)

### Why Cosine?

- Magnitude-independent — only direction matters
- Works well in high-dimensional spaces
- Computationally efficient

### Example

```python
cat = [0.8, 0.9, 0.1]
dog = [0.7, 0.85, 0.15]
king = [0.1, 0.2, 0.9]

cos(cat, dog) = 0.98  # Very similar!
cos(cat, king) = 0.32  # Unrelated
```

---

## 5. Vector Arithmetic and Analogies

### The Magic Property

Embeddings encode semantic relationships as geometric relationships:

```
king - man + woman ≈ queen
paris - france + germany ≈ berlin
walking - walked + went ≈ going
```

### How It Works

1. Take "king" vector
2. Subtract "man" vector (remove maleness)
3. Add "woman" vector (add femaleness)
4. Find nearest word → "queen"

### Mathematical Intuition

Embeddings learn semantic dimensions:
- Dimension 1: Royalty (king=high, cat=low)
- Dimension 2: Gender (man=high, woman=low)
- Dimension 3: Animacy (cat=high, stone=low)

Vector arithmetic manipulates these dimensions.

### Limitations

- Not perfect — nearest word isn't always correct
- Requires large corpora (billions of words)
- Some analogies work better than others
- Can reflect cultural biases

---

## 6. Visualizing Embeddings

### The Problem

Embeddings are 100-300 dimensional — can't visualize directly.

### Solution: Dimensionality Reduction

#### PCA (Principal Component Analysis)
- Linear projection
- Preserves maximum variance
- Fast and deterministic
- Preserves global structure

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Non-linear projection
- Preserves local structure
- Reveals clusters
- Slow, stochastic

#### UMAP (Uniform Manifold Approximation and Projection)
- Similar to t-SNE but faster
- Preserves both local and global structure
- Modern preference

### Usage

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 300D → 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, word in enumerate(vocab):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
```

---

## 7. Pre-trained Embeddings

### Popular Models

| Model | Organization | Training Data | Dimensions | Year |
|-------|-------------|---------------|------------|------|
| Word2Vec | Google | Google News (100B words) | 300 | 2013 |
| GloVe | Stanford | Wikipedia + Gigaword (6B tokens) | 50-300 | 2014 |
| FastText | Facebook | Common Crawl (600B tokens) | 300 | 2016 |

### Word2Vec vs GloVe

| Feature | Word2Vec | GloVe |
|---------|----------|-------|
| Method | Predictive (neural network) | Count-based (matrix factorization) |
| Training | Local context window | Global co-occurrence matrix |
| Speed | Faster | Slower |
| Performance | Similar | Similar |

### FastText Advantage

Uses character n-grams:
```
"running" → ["<ru", "run", "unn", "nni", "nin", "ing", "ng>"]
```

Benefits:
- Handles out-of-vocabulary words
- Better for morphologically rich languages (Turkish, Finnish)
- Robust to typos

---

## 8. Evaluating Embedding Quality

### 1. Intrinsic Evaluation

Test embeddings directly:

#### Word Similarity
- Datasets: WordSim-353, SimLex-999, RG-65
- Correlation with human ratings
- Metric: Spearman correlation

#### Analogy Task
- Google analogy dataset (19,544 questions)
- Format: a:b :: c:?
- Accuracy: Is nearest word correct?

### 2. Extrinsic Evaluation

Test on downstream tasks:
- Sentiment analysis
- Named entity recognition (NER)
- Text classification
- Question answering

Better embeddings → better task performance

### 3. Qualitative Analysis

- Inspect nearest neighbors
- Test analogies manually
- Visualize in 2D/3D
- Check clustering coherence

---

## 9. Limitations of Static Embeddings

### 1. Polysemy

Same word, different meanings → same vector

```
"I walked by the river bank" (bank = shore)
"I deposited money at the bank" (bank = financial institution)
→ Both get the same "bank" vector
```

### 2. Out-of-Vocabulary (OOV)

Words not seen during training get no embedding.

Solutions:
- FastText: Character n-grams
- Subword tokenization (BPE)
- Assign random vector (bad)

### 3. No Context

Same word, different contexts → same vector

```
"I love this movie" (love = enjoy)
"I love you" (love = romantic)
→ Same "love" vector
```

### 4. Fixed Vocabulary

Cannot adapt to new words after training.

### 5. Biases

Reflects biases in training data:
```
doctor - man + woman ≈ nurse
(gender bias)
```

---

## 10. Modern Alternatives: Contextual Embeddings

### The Problem

Static embeddings ignore context.

### Solution: Contextual Embeddings

Each token gets a different vector based on its context.

### ELMo (2018)

- Bidirectional LSTM
- Character-level input
- Combines embeddings from all layers

### BERT (2018)

- Transformer-based
- Bidirectional context
- Masked language modeling
- Sentence pair tasks

### GPT (2018-2024)

- Transformer-based
- Autoregressive (left-to-right)
- Causal language modeling
- Scaling laws

### Static vs Contextual

| Feature | Static (Word2Vec) | Contextual (BERT) |
|---------|-------------------|-------------------|
| Vectors per word | 1 | ∞ (context-dependent) |
| Polysemy | No | Yes |
| Context | No | Yes |
| Speed | Fast | Slow |
| Memory | Low | High |
| Usage | As features | Requires fine-tuning |

---

## 11. Practical Application

### When to Use Embeddings

1. **Static embeddings (Word2Vec, GloVe)**:
   - Fast prototyping
   - Limited compute
   - Simple tasks (similarity, clustering)
   - Feature engineering

2. **Contextual embeddings (BERT, GPT)**:
   - Production systems
   - Complex tasks (QA, NER)
   - Sufficient compute
   - State-of-the-art performance

### Code Example: Word2Vec with gensim

```python
from gensim.models import Word2Vec

# Prepare corpus
sentences = [['cat', 'sits', 'on', 'mat'],
             ['dog', 'runs', 'in', 'park']]

# Train model
model = Word2Vec(sentences, vector_size=100, window=5, 
                 min_count=1, workers=4, sg=1)  # sg=1: skip-gram

# Use
vec = model.wv['cat']  # Get embedding vector
similar = model.wv.most_similar('cat', topn=5)  # Similar words
result = model.wv.most_similar(positive=['king', 'woman'], 
                                negative=['man'])  # Analogy
```

### Code Example: Load Pre-trained Embeddings

```python
from gensim.models import KeyedVectors

# Load Word2Vec
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', 
                                           binary=True)

# Load GloVe (text format)
model = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', 
                                           binary=False, no_header=True)
```

---

## 12. Key Takeaways

1. **Embeddings encode meaning**: Dense vectors capture semantic relationships
2. **Context matters**: Distributional hypothesis is the foundation
3. **Vector arithmetic works**: Analogies are geometric relationships
4. **Visualization provides insight**: PCA/t-SNE reveal high-dimensional data
5. **Pre-trained > from scratch**: Use models trained on billions of words
6. **Static < contextual**: Modern LLMs use contextual embeddings
7. **Evaluation is important**: Use both intrinsic and extrinsic metrics
8. **Biases exist**: Embeddings reflect societal biases

---

## 13. What's Next: Attention Mechanisms

Embeddings convert words to vectors, but how do we process sequences?

Next: **Attention mechanisms** — allowing models to focus on different parts of the input.

```
"The animal couldn't cross the street because it was too wide"
→ What does "it" refer to? "animal" or "street"?
→ Attention solves this!
```

This is the foundation of Transformers and modern LLMs.

---

## Additional Resources

### Papers
- [Word2Vec Original Paper](https://arxiv.org/abs/1301.3781) — Efficient Estimation of Word Representations
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf) — Global Vectors for Word Representation
- [FastText Paper](https://arxiv.org/abs/1607.04606) — Enriching Word Vectors with Subword Information

### Tools and Libraries
- [gensim](https://radimrehurek.com/gensim/) — Python Word2Vec implementation
- [FastText](https://fasttext.cc/) — Facebook's official library
- [GloVe](https://nlp.stanford.edu/projects/glove/) — Stanford's pre-trained vectors

### Interactive Demos
- [Embedding Projector](https://projector.tensorflow.org/) — TensorFlow's 3D visualization tool
- [Word2Vec Visualizer](https://ronxin.github.io/wevi/) — Interactive embedding exploration
- [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) — Multilingual embeddings
