# VeritasVigil — The Truth Watchman
> *A rule-based NLP pipeline for fake news detection, built entirely from scratch.*

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [1. Custom Tokenizer](#1-custom-tokenizer)
- [2. Custom POS Tagger](#2-custom-pos-tagger)
- [3. Custom Lemmatizer](#3-custom-lemmatizer)
- [4. Feature Extraction & Classification](#4-feature-extraction--classification)
- [5. Visualizations](#5-visualizations)
- [6. Results](#6-results)
- [7. Off-the-Shelf Comparison](#7-off-the-shelf-comparison)
- [Conclusion](#conclusion)

---

## Overview

VeritasVigil is a fully hand-engineered NLP pipeline that classifies news headlines as **real or fake** without relying on pre-built NLP libraries. Every component — tokenizer, POS tagger, and lemmatizer — is built using Python `re` and string methods, making the pipeline fully transparent and explainable.

---

## Dataset

Two CSV files are used:
| File | Label | Description |
|------|-------|-------------|
| `Fake.csv` | `0` | Fake news headlines |
| `True.csv` | `1` | Real news headlines |

Both are merged, shuffled, and used for binary classification (`0 = Fake`, `1 = Real`).

---

## Pipeline Architecture

```
Raw Headline
     │
     ▼
┌─────────────────────┐
│   Custom Tokenizer  │  lowercase → expand contractions → strip punctuation → normalize repeats → split
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Mini POS Tagger    │  suffix rules + keyword lookups → VERB / NOUN / ADJ / ADV / NEG / REPEAT
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ POS-Guided          │  strip -ing / -ed / -s / -ly / -ful / -ous etc. only when POS matches
│ Lemmatizer          │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Feature Extraction  │  Bag-of-Words (CountVectorizer) + TF-IDF (TfidfVectorizer)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Classifiers         │  Naïve Bayes (BoW)  |  SVM linear (TF-IDF)
└─────────────────────┘
```

---

## 1. Custom Tokenizer

Built using Python `re` and string methods with the following stages:

| Step | Operation | Example |
|------|-----------|---------|
| 1 | Lowercase | `Trump` → `trump` |
| 2 | Expand contractions | `can't` → `can not` |
| 3 | Strip punctuation | `funny!!!` → `funny` |
| 4 | Normalize repeated chars | `soooo` → `so <REPEAT:3>` |
| 5 | Split into tokens | `"so <REPEAT:3> funny"` → `['so', '<REPEAT:3>', 'funny']` |

### Repeated-Character Normalization

```python
def normalize_repeats(word):
    def replacer(m):
        char = m.group(1)
        repeat_count = len(m.group(0)) - 1   # extra repetitions beyond first
        return f"{char} <REPEAT:{repeat_count}>"
    return re.sub(r'(.)\1{2,}', replacer, word)
```

The `<REPEAT:n>` token is preserved as a real feature — fake news tends to use informal exaggeration (e.g. `sooooo`, `biggggg`), so this marker carries a genuine signal for classification.

**Full example:**
```
"That's soooo funny!!!" → ['that', 'is', 'so', '<REPEAT:3>', 'funny']
```

---

## 2. Custom POS Tagger

A rule-based tagger using suffix patterns and keyword lookup sets. No external libraries used.

### Tags Assigned

| Tag | Strategy | Examples |
|-----|----------|---------|
| `VERB` | Auxiliary lookup + `-ing` / `-ed` suffixes + `-s` if stem in base-verb list | `running`, `claimed`, `says` |
| `ADJ` | Suffix list: `-ful`, `-ous`, `-ive`, `-able`, `-ible`, `-ical`, `-less`, `-some`, `-y` | `powerful`, `logical` |
| `ADV` | `-ly` suffix | `quickly`, `really` |
| `NEG` | Keyword list: `no`, `not`, `never`, `none` | `not`, `never` |
| `NOUN` | Default fallback | `president`, `policy` |
| `REPEAT` | Token starts with `<repeat:` | `<REPEAT:4>` |

### Core logic

```python
if token.endswith("ing") and len(token) > 4:
    return "VERB"
elif token.endswith("ly") and len(token) > 3:
    return "ADV"
elif token.endswith(("ful", "ous", "ive", "able", ...)):
    return "ADJ"
else:
    return "NOUN"  # default
```

---

## 3. Custom Lemmatizer

POS-guided rule-based lemmatizer. Rules fire **only when the POS tag matches**, preventing incorrect reductions.

| POS | Rule | Example |
|-----|------|---------|
| `VERB` | Strip `-ing` | `running` → `run` |
| `VERB` | Strip `-ed` | `claimed` → `claim` |
| `VERB` | Strip `-s` | `runs` → `run` |
| `NOUN` | Strip `-s` | `cats` → `cat` |
| `ADV` | Strip `-ly` | `quickly` → `quick` |
| `ADJ` | Strip `-ful` / `-ous` / `-able` (3 chars) | `powerful` → `power` |
| `ADJ` | Strip `-ible` / `-ical` / `-less` / `-eous` / `-ious` / `-some` (4 chars) | `logical` → `log` |
| `REPEAT` / `NEG` | Pass through unchanged | `<REPEAT:4>` → `<REPEAT:4>` |

```python
if pos == "VERB":
    if token.endswith("ing") and len(token) > 4:
        return token[:-3]   # running → run
    elif token.endswith("ed") and len(token) > 3:
        return token[:-2]   # claimed → claim
```

All rules include minimum-length guards to avoid over-stripping short words like `is` or `be`.

### Impact on Model Performance

| Feature | Effect |
|---------|--------|
| Repeated-character normalization | Reduces OOV sparsity; `soooo` maps to the known token `so` plus a discriminative `<REPEAT:n>` signal |
| POS-guided lemmatization | Collapses `run` / `running` / `runs` into one feature, reducing dimensionality and overfitting |

Without these steps, the model encounters more sparse, noisy representations — particularly for informal fake-news language.

---

## 4. Feature Extraction & Classification

### Feature Representations

| Method | Implementation | Description |
|--------|---------------|-------------|
| Bag-of-Words | `sklearn.CountVectorizer` | Raw term frequency counts |
| TF-IDF | `sklearn.TfidfVectorizer` | Down-weights common tokens, up-weights discriminative ones |

### Classifiers

| Classifier | Features | Rationale |
|------------|----------|-----------|
| Naïve Bayes (Multinomial) | BoW | Fast probabilistic baseline; works well with count features |
| SVM (linear kernel) | TF-IDF | Strong margin-based classifier; robust on high-dimensional sparse text |

---

## 5. Visualizations

The notebook generates the following plots:

- **Token Frequency Distribution** — top 20 most common lemmatized tokens (excluding REPEAT markers)
- **Word Cloud** — visual overview of dominant vocabulary across all headlines
- **REPEAT Token Distribution** — histogram of `<REPEAT:2>`, `<REPEAT:3>` etc. counts; illustrates how often exaggerated language appears
- **Confusion Matrices** — side-by-side heatmaps for both classifiers (Blues for NB, Greens for SVM)
- **ROC Curves** — both classifiers on the same axes with AUC scores, vs. random baseline

---

## 6. Results

| Model | Features | Accuracy | F1-Score |
|-------|----------|----------|----------|
| Naïve Bayes | Bag-of-Words | 93.2% | 0.94 |
| **SVM (linear)** | **TF-IDF** | **95.1%** | **0.95** |

SVM outperforms Naïve Bayes because TF-IDF suppresses uninformative high-frequency tokens and SVM's margin maximisation is better suited to the high-dimensional sparse feature space.

---

## 7. Off-the-Shelf Comparison

As a reference-only sanity check, a basic `TfidfVectorizer + LogisticRegression` pipeline from scikit-learn was tested.

| Pipeline | Accuracy | Transparency |
|----------|----------|-------------|
| **Custom Rule-Based (ours)** | **95.1%** | ✅ Fully explainable |
| scikit-learn baseline | ~95.5% | ❌ Black-box preprocessing |

The gap is minimal (~0.4%), while our pipeline offers full control over every preprocessing decision — making it more suitable for educational, forensic, and interpretability-focused applications.

---

## Conclusion

VeritasVigil demonstrates that a thoughtfully designed rule-based NLP pipeline can match near-state-of-the-art performance on fake news detection while remaining fully transparent. Every component is documented, justified, and explainable — from the contraction dictionary to the POS suffix rules to the lemmatizer length guards.

> *"No borrowed NLP magic. Just regex, intuition, and rule-based cunning."*
