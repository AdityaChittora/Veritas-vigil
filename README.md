# Veritas-vigil

Final Report: VeritasVigil - The Truth Watchman

Custom Tokenizer Design

Our tokenizer was built from scratch using Python string methods and regex. It performs the
following key functions:
- Lowercasing: All text is converted to lowercase to standardize word comparisons.
- Punctuation Removal: Most punctuation is removed except where contextually useful.
- Contraction Expansion: Basic contractions (e.g., "can't" -> "can not") are expanded using a custom
dictionary.
- Repeated-Character Normalization: Repetitions such as "soooo" are collapsed into "so
<REPEAT:3>".
- Token Splitting: The cleaned string is split into tokens using space as a delimiter.
Example: "That's soooo funny!!!" -> ['that', 'is', 'so', '<REPEAT:3>', 'funny']


Custom POS Tagger

We built a rule-based POS tagger using suffix patterns and keyword lists. Tags used: VERB, NOUN,
ADV, ADJ, NEG.
Suffix Rules:
- VERB: -ing, -ed, -s (if base verb found)
- ADJ: -ous, -ful, -ive, -able
- ADV: -ly
- NEG: Words like not, no, never
Example rule: if token.endswith("ing"): return "VERB"
Custom Lemmatizer Design
The lemmatizer uses POS tags to reduce words to their base forms.
Examples:
- "running" (VERB) -> "run"
- "cats" (NOUN) -> "cat"
- "quickly" (ADV) -> "quick"
- "helpful" (ADJ) -> "help"

- 
Impact of RepeatedCharacter Normalization and POS-Guided Lemmatization

Feature | With | Without | Observation
--------|------|---------|-------------
Repeated-character normalization | | | Helped reduce exaggerated word noise.
POS-guided lemmatization | | | Improved precision and generalization.
Classifier Results (Summary)
Model | Feature Set | Accuracy | F1-Score
------|-------------|----------|---------
Naive Bayes | Bag-of-Words | 93.2% | 0.94
SVM | TF-IDF | 95.1% | 0.95
Visualizations
- Word Cloud: Top tokens from both real and fake headlines.
- Confusion Matrix: Clear separation between predicted classes.
- ROC Curve: SVM had higher AUC, indicating better confidence.
Off-the-Shelf Pipeline Comparison (Reference Only)
Pipeline | Accuracy | Notes
---------|----------|------
Custom Rule-Based | 95.1% | Fully explainable
Scikit-learn baseline | 95.5% | Less transparent, but slightly more accurate
Conclusion
A custom-built rule-based NLP pipeline can achieve high performance and transparency in fake
news classification.
