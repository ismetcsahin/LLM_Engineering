# Day 2: Tokenization and Vocabulary - Methods Overview

## Introduction

This document covers how language models convert raw text into numbers. Unlike humans who read words, models process **tokens** — sub-word units produced by algorithms like Byte Pair Encoding (BPE). Understanding tokenization is essential for controlling costs, context length, and model behavior.

---

## 1. What Is a Token?

### Definition
A token is the basic unit of text that a language model processes. It can be:
- A full word: `"hello"` → `["hello"]`
- A sub-word: `"tokenization"` → `["token", "ization"]`
- A character: `"X"` → `["X"]`
- Punctuation or whitespace: `","`, `" "`

### Why Not Just Use Words?
- **Unknown words**: Word-level models fail on new words
- **Morphology**: "run", "running", "runner" share meaning — sub-words capture this
- **Vocabulary size**: Character-level is too granular; word-level is too large
- **Sub-word balance**: BPE finds the sweet spot

---

## 2. Byte Pair Encoding (BPE)

### What It Is
A data compression algorithm adapted for NLP. It iteratively merges the most frequent adjacent symbol pairs to build a vocabulary.

### Algorithm Steps
1. Start with character-level vocabulary (each character is a token)
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until vocabulary reaches target size

### Example
```
Corpus: "low lower lowest"
Initial: l o w </w>, l o w e r </w>, l o w e s t </w>

Step 1: merge (l, o) → lo
Step 2: merge (lo, w) → low
Step 3: merge (e, r) → er
...
Final tokens: low, lower, lowest, er, est
```

### Why It Works
- Common words become single tokens (efficient)
- Rare words are split into known sub-parts (no unknown tokens)
- Vocabulary size is controllable

### Limitations
- Greedy algorithm — not globally optimal
- Language-agnostic merges may be suboptimal for some languages
- Token boundaries don't always align with linguistic morphemes

---

## 3. tiktoken — OpenAI's Tokenizer

### What It Is
A fast BPE tokenizer library by OpenAI, used in production for GPT models.

### Available Encodings

| Encoding | Models | Vocabulary Size |
|----------|--------|----------------|
| `gpt2` | GPT-2 | 50,257 |
| `cl100k_base` | GPT-3.5, GPT-4 | 100,277 |
| `o200k_base` | GPT-4o | 200,019 |

### Basic Usage
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")   # → list of token IDs
text = enc.decode([9906, 11, 1917, 0]) # → "Hello, world!"
```

### Key Insight
Larger vocabularies mean more text fits in fewer tokens — better efficiency and longer effective context.

---

## 4. Multilingual Token Costs

### The Problem
BPE vocabularies are trained predominantly on English text. This means:
- English: ~1 token per word (very efficient)
- Turkish, Finnish: agglutinative languages → more tokens per word
- Chinese, Japanese, Arabic: non-Latin scripts → often 1–3 chars per token

### Why It Matters
- **API cost**: More tokens = higher cost for the same meaning
- **Context window**: Non-English text fills context faster
- **Fairness**: Models trained on English-heavy data perform better in English

### Chars per Token Benchmark (cl100k_base)
```
English:  ~4.0 chars/token  (most efficient)
Spanish:  ~3.5 chars/token
Turkish:  ~3.0 chars/token
Russian:  ~2.0 chars/token
Arabic:   ~1.5 chars/token
Chinese:  ~1.5 chars/token  (least efficient)
```

### Real-World Impact

For a 1000-word document:
- English: ~750 tokens → $0.00375 (gpt-4o input)
- Turkish: ~1200 tokens → $0.00600 (gpt-4o input)
- Chinese: ~1500 tokens → $0.00750 (gpt-4o input)

**Non-English languages can cost 60-100% more for the same semantic content.**

---

## 5. Special Tokens

### What They Are
Reserved tokens with special meaning that control model behavior.

### Common Special Tokens

| Token | Purpose |
|-------|---------|
| `<\|endoftext\|>` | Marks end of a document |
| `<\|fim_prefix\|>` | Fill-in-the-middle: prefix |
| `<\|fim_middle\|>` | Fill-in-the-middle: middle |
| `<\|fim_suffix\|>` | Fill-in-the-middle: suffix |

### Chat Templates
Chat models wrap messages with role markers and special tokens:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is BPE?
<|im_end|>
```
Each message adds ~4 overhead tokens beyond the content.

---

## 6. Token Counting for API Cost Estimation

### Formula
```
Total Cost = (Input Tokens / 1,000,000) × Input Price
           + (Output Tokens / 1,000,000) × Output Price
```

### GPT-4o Pricing (2024, per 1M tokens)
| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $5.00 | $15.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |

### Practical Tips
- Count tokens **before** sending requests to avoid surprises
- System prompts are charged on every request — keep them concise
- Output tokens cost 3× more than input tokens (gpt-4o) — be specific in prompts

---

## 7. Tokenization Edge Cases

### Numbers
Large numbers are split into multiple tokens, making arithmetic harder:
```
"100"    → 1 token
"10000"  → 2 tokens: ["100", "00"]
"99999"  → 3 tokens: ["999", "99"]
```

### Whitespace
Leading spaces matter — they change the token:
```
"hello"  → ["hello"]
" hello" → [" hello"]  ← different token ID!
```

### Capitalization
```
"hello" → 1 token
"Hello" → 1 token (different ID)
"HELLO" → 1 token (different ID)
```

### Code
Code tokenizes efficiently because keywords are common in training data:
```python
"def hello_world():"  → ["def", " hello", "_world", "():"]
```

---

## 8. Vocabulary Size Evolution

| Model | Year | Vocab Size | Key Improvement |
|-------|------|-----------|----------------|
| GPT-2 | 2019 | 50,257 | Baseline BPE |
| GPT-3.5/4 | 2022–23 | 100,277 | Better multilingual coverage |
| GPT-4o | 2024 | 200,019 | Efficient non-Latin scripts |

Larger vocabularies reduce token count for the same text, effectively doubling the usable context window.

---

## 9. Limitations of Current Tokenization

### 1. Arithmetic Failures
Numbers split across tokens make simple math unreliable. Models must reason across token boundaries.

### 2. Language Inequality
Non-English languages use more tokens for equivalent meaning → higher cost, shorter effective context.

### 3. Tokenization Artifacts
- Rare or adversarial tokens can cause unexpected model behavior
- The "SolidGoldMagikarp" problem: tokens that appear in vocabulary but never in training text cause glitches

### 4. No Linguistic Awareness
BPE is purely statistical — it doesn't know about morphemes, syllables, or word roots.

### 5. Fixed Vocabulary
Once trained, the tokenizer cannot adapt to new domains without retraining.

---

## 10. What's Next: From Tokens to Embeddings

Tokenization converts text → token IDs (integers). The next step is converting those IDs into **dense vectors** (embeddings) that capture semantic meaning.

```
"cat" → token ID 9246 → embedding vector [0.23, -0.41, 0.87, ...]
"dog" → token ID 5765 → embedding vector [0.21, -0.38, 0.91, ...]
# Similar meaning → similar vectors
```

This is the bridge between raw text and the mathematical world of neural networks.

---

## Comparison: Tokenization Strategies

| Strategy | Unit | Vocab Size | OOV Handling | Example |
|----------|------|-----------|-------------|---------|
| Character | Single char | ~100 | None | `h,e,l,l,o` |
| Word | Full word | 50K–1M | Fails | `hello` |
| BPE | Sub-word | 32K–200K | Sub-parts | `hel,lo` |
| WordPiece | Sub-word | 30K | Sub-parts | `hell,##o` |
| SentencePiece | Sub-word | Configurable | Sub-parts | `▁hello` |

---

## Practical Takeaways

1. **Always count tokens before API calls** — use `tiktoken` to estimate cost
2. **Shorter prompts = lower cost** — remove unnecessary context
3. **Non-English prompts cost more** — factor this into multilingual app budgets
4. **Output tokens are expensive** — instruct the model to be concise
5. **Vocabulary size matters** — newer models handle non-Latin scripts better

## Additional Resources and Tools

### Tokenizer Comparison Tools
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Test your text across different models
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers) — Various tokenizer implementations
- [tiktokenizer](https://tiktokenizer.vercel.app/) — Visual tokenizer explorer

### Further Reading
- [BPE Original Paper](https://arxiv.org/abs/1508.07909) — Neural Machine Translation of Rare Words
- [SentencePiece](https://github.com/google/sentencepiece) — Google's language-agnostic tokenizer
- [tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI's official library
- [The SolidGoldMagikarp Problem](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/) — Understanding tokenization artifacts
