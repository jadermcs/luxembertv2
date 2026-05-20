# EVALUATION — Benchmarks for a Luxembourgish Encoder

An encoder is judged by the quality of its **representations**, so almost all evaluation is
*fine-tune (or probe) on a downstream task* + a couple of intrinsic signals. Luxembourgish is
low-resource, so the benchmark set is small — use **all** of it and **report mean ± std over
multiple seeds** (variance is high with little data).

## Evaluation protocol
- **Full fine-tuning** and **linear probing** (frozen encoder) — probing isolates representation
  quality; fine-tuning shows end-task ceiling.
- Also test **few-shot fine-tuning** (e.g. 10/50/100 examples) — the realistic low-resource setting.
- Baselines to always include: **mBERT**, **XLM-R**, **mDeBERTa-v3**, **LuxemBERT**, and a
  German BERT (DE is the closest high-resource neighbour).
- Aggregate into a single **GLUE-style mean score** for ranking, but keep per-task tables.

---

## 1. Token-level / structured tasks

| Task | Dataset | Metric | Notes |
|---|---|---|---|
| **POS tagging** | **LuxBank** (UD treebank, first for LB); LuxemBERT POS set | Accuracy / F1 | UD tagset; core syntactic probe |
| **Dependency parsing** | **LuxBank** (Universal Dependencies) | UAS / LAS | needs LuxBank's full annotations |
| **NER** | **WikiANN / PAN-X** (`lb`); **LuxemBERT NER** set | span-F1 | WikiANN is silver/Wikipedia-derived |

- LuxBank: <https://arxiv.org/abs/2411.04813>
- WikiANN/PAN-X: <https://huggingface.co/datasets/unimelb-nlp/wikiann> (config `lb`)

---

## 2. Sentence / document classification (NLU)

| Task | Dataset | Metric | Notes |
|---|---|---|---|
| **Topic classification** | **SIB-200** (`ltz_Latn`) | Accuracy / F1 | FLORES-derived, clean, comparable cross-lingually |
| **News classification** | **LuxemBERT** news set | Accuracy | LB-native domain data |
| **Text classification** | **Taxi1500** (`lb` if covered) | F1 | Bible-domain, 1500-lang coverage |
| **NLI / Winograd** | **LuxemBERT LNLI** (LB Winograd-style NLI) | Accuracy | tests reasoning over LB |
| **Sentiment analysis** | LB sentiment set (LuhMe 2024, "Mapping Sentiments") | Acc / macro-F1 | binary + 3-class |

- SIB-200: <https://huggingface.co/datasets/Davlan/sib200>
- LuxemBERT tasks + data: <https://github.com/Trustworthy-Software/LuxemBERT>
- Sentiment (LuhMe 2024): <https://aclanthology.org/2024.luhme-1.3.pdf>

---

## 3. Reading comprehension

| Task | Dataset | Metric | Notes |
|---|---|---|---|
| **Multiple-choice MRC** | **Belebele** (`ltz_Latn`) | Accuracy | 4-way; hard for encoders, run as span/answer-selection or with a classification head |

- Belebele: <https://huggingface.co/datasets/facebook/belebele>

---

## 4. Embedding quality / retrieval (if you want a sentence encoder)

| Task | Dataset | Metric | Notes |
|---|---|---|---|
| **Bitext mining / sentence retrieval** | **FLORES-200** (`ltz_Latn`), **Tatoeba** (`lb`) | top-1 accuracy / F1 | LB↔DE/EN/FR alignment quality of mean-pooled embeddings |
| **Semantic similarity** | translate STS or use FLORES pairs | Spearman | no native LB STS; cross-lingual proxy |

- FLORES-200: <https://huggingface.co/datasets/facebook/flores>
- Tatoeba: <https://huggingface.co/datasets/Helsinki-NLP/tatoeba>

---

## 5. Intrinsic / diagnostic signals (cheap, run continuously)

- **Pseudo-perplexity** of the MLM head on a held-out LB set (compare checkpoints/objectives).
- **MLM / RTD accuracy** on held-out data.
- **Tokenizer fertility** (tokens/word) on held-out LB text — affects everything downstream.
- **Loss-vs-tokens** curves to read marginal value of extra epochs (data-constrained regime).
- **Text normalization** (LB spelling-variation → standard) as an auxiliary diagnostic, if you
  build a head for it. Dataset: <https://arxiv.org/abs/2412.09383>.

---

## Recommended minimal scorecard (start here)
1. **POS** (LuxBank) — Accuracy
2. **NER** (WikiANN `lb`) — span-F1
3. **Topic** (SIB-200 `ltz_Latn`) — Accuracy
4. **NLI** (LuxemBERT LNLI) — Accuracy
5. **Sentiment** (LuhMe 2024) — macro-F1
6. **Belebele** (`ltz_Latn`) — Accuracy
7. Intrinsic: pseudo-perplexity + tokenizer fertility

Report each as mean ± std over ≥3 seeds, plus a single averaged rank across tasks.

## Caveats
- Several LB benchmarks are **tiny** → noisy; prefer multiple seeds and report variance.
- WikiANN/Taxi1500 are **silver/auto-derived** → treat as indicative, not gold.
- Belebele/SIB-200/FLORES are **FLORES-translated** → cross-lingually comparable but not native
  domain; pair them with native LB sets (LuxemBERT, LuxBank, sentiment) for a fair picture.
</content>
