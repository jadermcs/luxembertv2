# IDEAS — Data-Efficient Luxembourgish Encoder

**Goal:** train an encoder (BERT-style) for Luxembourgish (LB).
**Corpus:** [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
subset `ltz_Latn` ≈ **358K documents / ~496 MB** (~50–80M words). This is genuinely
**low-resource**, so every choice below is ranked by *value per token of data*.

> **Guiding principle:** with so little native data, the two biggest levers are
> (1) **cross-lingual transfer from German** (LB is a Moselle-Franconian variety close to DE)
> and (2) a **sample-efficient pretraining objective** (RTD/ELECTRA-style). Architecture and
> optimizer tweaks are second-order — test them, but expect smaller gains.

Legend: 🔥 high expected impact · ⚙️ medium · 🧪 exploratory.

---

## A. Pretraining objective (biggest single lever)

- 🔥 **ELECTRA (Replaced Token Detection)** — discriminate real vs. generator-replaced tokens.
  Learns from *all* positions, not just 15% masked → 3–4× more sample-efficient than MLM. Ideal
  for low data. (Clark et al. 2020)
- 🔥 **DeBERTa-v3 (RTD + Gradient-Disentangled Embedding Sharing)** — current SOTA for
  sample-efficient encoders; combines disentangled attention with ELECTRA-style RTD. Strong
  default to beat. (He et al. 2021)
- ⚙️ **RoBERTa MLM** — robust baseline; dynamic masking, no NSP. Use as the control everything
  is measured against.
- ⚙️ **Masking strategy sweep**: whole-word masking, **SpanBERT** span masking, **PMI/n-gram
  masking**. Span/WWM help morphologically rich, low-data languages.
- ⚙️ **Higher mask rate** (30–40%) — recent work shows >15% is often better, especially with
  larger models / RTD generators. (Wettig et al. 2023)
- 🧪 **Combined objectives**: MLM+RTD multitask, or **MASS/T5-style** denoising if you want a
  encoder-decoder variant later.
- 🧪 **Contrastive sentence objective** (SimCSE-style dropout, or TSDAE) as auxiliary loss if a
  *sentence* encoder is a goal.

---

## B. Cross-lingual transfer & data efficiency (most important for LB)

- 🔥 **Warm-start from German** — initialize from a German BERT/DeBERTa and continue pretraining
  on LB. LB shares huge lexical/syntactic overlap with DE. (cf. LuxemBERT "Lb/De" setup)
- 🔥 **Vocabulary / embedding transfer** when reusing a foreign model with a new LB tokenizer:
  **WECHSEL**, **FOCUS**, **OFA**, or **trans-tokenization**. Lets you keep pretrained weights
  while adopting an LB-optimal vocab.
- 🔥 **Data augmentation by partial translation from German** — LuxemBERT's core trick: cheaply
  "translate" DE text toward LB to multiply training data. Simple, large gains in their paper.
- 🔥 **Multilingual co-training**: pretrain on LB **+ German (+ French)** jointly, upsampling LB.
  LB borrows heavily from both; neighbours regularize and fill gaps.
- ⚙️ **Continue-pretrain a multilingual base** (XLM-R, mDeBERTa-v3, **Glot500**) on LB rather
  than training from scratch — strong baseline, very cheap.
- ⚙️ **Knowledge distillation** from a large multilingual teacher (XLM-R-large / mDeBERTa) into a
  small LB student — transfers signal that LB data alone can't.
- ⚙️ **Repeat data for multiple epochs** — in the data-constrained regime, repeating up to ~4
  epochs is nearly as good as fresh tokens. (Muennighoff et al. 2023, "Scaling
  Data-Constrained LMs") → plan for many epochs, not one pass.
- 🧪 **Curriculum learning**: DE→mixed→LB, or easy→hard by length/perplexity.
- 🧪 **Back-translation / round-trip augmentation** via an MT model with LB support.

---

## C. Architecture

- 🔥 **DeBERTa disentangled attention** — separate content/position attention + relative positions;
  consistently strong and pairs with RTD (see DeBERTa-v3 in §A).
- ⚙️ **ModernBERT recipe** — RoPE positions, GeGLU, alternating local/global attention, no
  bias terms, unpadding/sequence-packing. Modern, efficient, good defaults. (Warner et al. 2024)
- ⚙️ **RoPE** vs **ALiBi** vs **relative (T5/DeBERTa)** positional encodings — sweep; affects
  length generalization on short LB web text.
- 🔥 **AttnRes — Attention Residuals** (Moonshot AI / Kimi, 2026) — **drop-in replacement for
  fixed residual connections**: instead of blindly summing every previous layer's output, each
  layer does *content-aware, attention-based mixing* over its layer history. Reported ~25% better
  compute efficiency plus better deep-net info flow and training stability → fewer tokens to
  target loss, which is exactly what a low-data run needs. Cheap to try, applies to any of the
  backbones above. (<https://nerdschalk.com/moonshot-ais-attention-residuals-for-kimi-could-change-how-ai-models-use-layers/>)
- 🔥 **XSA — Exclusive Self-Attention** (Apple, Zhai, 2026, arXiv:2603.09078) — a **two-line
  change that stops a token from attending to its own position**, constraining attention to the
  component orthogonal to the token's own value. Frees attention capacity (the self/point-wise
  path is already covered by the FFN residual) → consistent gains across sizes up to 2.7B, larger
  as sequences grow. Near-zero cost, stacks with AttnRes and any objective.
  (<https://arxiv.org/abs/2603.09078>)
- 🧪 **GeGLU/SwiGLU FFN** vs vanilla GELU — small but consistent gains.
- 🧪 **Depth-vs-width** and **small-model bias** — with little data, *smaller* models (fewer
  params, more epochs) often generalize better; sweep param count against the data budget.

---

## D. Optimization

- 🔥 **Muon optimizer** — orthogonalized momentum updates (Newton–Schulz) for 2D weights, AdamW
  for the rest. Reported faster convergence / fewer tokens to target loss → directly useful when
  tokens are scarce. Test vs AdamW head-to-head. (Jordan et al. 2024)
- ⚙️ **AdamW** — baseline; tune β₂ (0.95–0.98), weight decay, ε.
- ⚙️ **LR schedule: WSD (warmup–stable–decay) / trapezoidal** — enables clean intermediate
  checkpoints and easy "add more epochs" without re-warming. Compare to cosine.
- ⚙️ **Large-batch training (LAMB)** if you scale batch size.
- 🧪 **Sophia** / second-order-ish optimizers — another fast-convergence candidate.
- 🧪 **z-loss / logit regularization** for stability at higher mask rates and small data.
- 🧪 **Weight averaging (EMA / model soups)** across checkpoints — cheap variance reduction.

---

## E. Tokenization (punches above its weight in low-resource)

- 🔥 **Train an LB-specific tokenizer** — SentencePiece **Unigram** vs **BPE**; pick by lowest
  **fertility** (tokens/word) on held-out LB. Lower fertility = more effective context & data.
- 🔥 **Vocab-size sweep** (16k / 32k / 48k) — smaller vocab can be better with little data
  (fewer rare embeddings to learn).
- ⚙️ **German-derived / shared DE+LB vocab** to maximize transfer (pairs with §B warm-start).
- ⚙️ **Byte-level fallback** to robustly handle LB diacritics (ë, é, ä) and code-switching
  (FR/DE/EN) common in LB web text.
- 🧪 **Morphology-aware** pre-tokenization for LB compounding.

---

## F. Regularization & training tricks

- 🔥 **Sequence packing / unpadding** — pack short LB web docs to remove pad waste; large
  throughput win → more effective epochs per GPU-hour.
- ⚙️ **Dropout / stochastic depth** tuning — more regularization helps in the small-data regime.
- ⚙️ **Gradient clipping** + careful warmup for stability.
- 🧪 **Token dropping** (skip easy tokens in early layers) for cheaper pretraining.
- 🧪 **Data filtering/dedup of `ltz_Latn`** + LB language-ID re-check — fineweb-2 is web text;
  cleaning and near-dedup can matter more than model tweaks. Consider
  [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ)-style model-based selection.

---

## G. Scaling / model-size strategy

- ⚙️ **Compute-/data-optimal sizing** — use data-constrained scaling laws to pick params given
  ~50–80M words; don't over-parameterize.
- 🧪 Ship **two sizes** (e.g. ~30M "small" and ~110M "base") so downstream users can trade
  speed vs accuracy.

---

## Suggested experiment order (ablate one axis at a time)
1. **Baseline:** RoBERTa-MLM from scratch, LB tokenizer (32k) — establishes the floor.
2. **Objective:** swap to **DeBERTa-v3 / ELECTRA-RTD** (expect the biggest jump).
3. **Transfer:** warm-start from **German** + vocab transfer (WECHSEL/FOCUS).
4. **Augmentation:** add **partial-translation** DE→LB data + multilingual co-training.
5. **Optimizer:** **Muon** vs AdamW on the best config.
6. **Architecture:** ModernBERT recipe / RoPE, then the cheap drop-in attention wins
   (**AttnRes**, **XSA**) on top.
7. **Tokenizer & size** sweeps to finalize.

Hold the eval suite fixed (see `EVALUATION.md`) and report mean ± std over ≥3 seeds for each
change.

---

### References (quick links)
- ELECTRA: <https://arxiv.org/abs/2003.10555>
- DeBERTa / DeBERTa-v3: <https://arxiv.org/abs/2006.03654> · <https://arxiv.org/abs/2111.09543>
- ModernBERT: <https://arxiv.org/abs/2412.13663>
- AttnRes — Attention Residuals (Kimi/Moonshot): <https://nerdschalk.com/moonshot-ais-attention-residuals-for-kimi-could-change-how-ai-models-use-layers/>
- XSA — Exclusive Self-Attention (Apple): <https://arxiv.org/abs/2603.09078>
- Muon optimizer: <https://kellerjordan.github.io/posts/muon/>
- Should You Mask 15%?: <https://arxiv.org/abs/2202.08005>
- Scaling Data-Constrained LMs: <https://arxiv.org/abs/2305.16264>
- WECHSEL: <https://arxiv.org/abs/2112.06598> · FOCUS: <https://arxiv.org/abs/2305.14481>
- LuxemBERT (DE-augmentation for LB): <https://aclanthology.org/2022.lrec-1.543.pdf>
</content>
