# News Text Classification: Classical ML vs. LLMs

A comparative study of **TF-IDF + Logistic Regression** vs. **Qwen 1.5B-Chat** (zero-shot, few-shot, and LoRA fine-tuned) on the **AG News** dataset for multi-class news article classification.

> **Dataset:** [AG News (Hugging Face)](https://huggingface.co/datasets/ag_news)  
> **Fine-tuned Model:** [ecettt/qwen-1.5b-finetuned](https://huggingface.co/ecettt/qwen-1.5b-finetuned)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
  - [TF-IDF + Logistic Regression](#1-tf-idf--logistic-regression)
  - [Qwen 1.5B-Chat (Prompt Engineering)](#2-qwen-15b-chat-prompt-engineering)
  - [Qwen 1.5B LoRA Fine-Tuning](#3-qwen-15b-lora-fine-tuning)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

This project evaluates classical machine learning methods against modern large language model (LLM) approaches for structured text classification. We systematically compare three pipelines on the AG News benchmark, exploring the trade-offs between interpretability, computational efficiency, and accuracy.

---

## Dataset

**AG News** — 127,600 news article samples from 2,000+ sources.

| Split | Samples |
|-------|---------|
| Train (full) | 120,000 |
| Test | 7,600 |
| Train split (`train_split.csv`) | 108,000 (90% of train) |
| Validation (`validation.csv`) | 12,000 (10% of train) |
| Train subsample | 16,000 |
| Validation subsample | 4,000 |

**Classes:**

| Label | Category |
|-------|----------|
| 0 | 🌍 World |
| 1 | 🏅 Sports |
| 2 | 💼 Business |
| 3 | 🔬 Sci/Tech |

- **No missing values** — all texts and labels are complete.
- **Average article length:** 236 characters (75th percentile: 266 characters).
- **Class distribution:** Balanced across all four categories.

> Articles are short enough to use in full for both classical ML and LLM-based approaches without truncation concerns.

---

## Methods

### 1. TF-IDF + Logistic Regression

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into numerical feature vectors by weighting words based on their frequency in a document relative to their rarity across all documents. This highlights informative, discriminative terms while downweighting common stop words.

**Logistic Regression** learns a linear decision boundary using the TF-IDF features, assigning each article to the class with the highest predicted probability.

**Key findings:**
- Best performance at `max_features=40,000`.
- Custom text preprocessing (removing digits and special characters) *hurt* accuracy (92.14% → 91.60%), as it stripped informative tokens like `Q2`, `3-0`, `NASA`, `COVID-19`.
- Final decision: **no custom text cleaning** applied before training.

| Dataset | Accuracy |
|---------|----------|
| Validation | **92.14%** |
| Test | **91.67%** |

---

### 2. Qwen 1.5B-Chat (Prompt Engineering)

**Qwen 1.5B-Chat** is a Transformer-based LLM (500M parameters) developed by Alibaba, optimized for instruction-following. Classification is performed via text generation — the model generates a category name given a structured prompt.

The model uses **greedy decoding** (`max_new_tokens=20`) and outputs are post-processed to extract category labels. Invalid outputs are labeled `-1` and excluded from evaluation.

**Prompt engineering attempts:**

| Attempt | Strategy | Result |
|---------|----------|--------|
| 1 | Zero-shot (default) | 47% accuracy; low F1 for World & Sci/Tech |
| 2 | Detailed label descriptions | Slight improvement in World & Sci/Tech |
| 3 | Few-shot with misclassified keywords | Improved accuracy |
| 4 | Few-shot on only 2 labels (experimental) | **Worst performance** |
| 5 (Final) | Few-shot for all labels, simplified for low-F1 classes | **51% accuracy — best prompt-based result** |
| 5b | Threshold hinting for World recall | No improvement; model weight adjustment needed |

**Limitations identified:**
- Ambiguous label boundaries (World vs. Business overlap, unrelated Sci/Tech labels)
- Inefficient probability threshold for World news
- Dataset contains recent events (e.g., Gaza) not well represented in the model's training data
- General architectural limitations compared to larger LLMs

---

### 3. Qwen 1.5B LoRA Fine-Tuning

Since prompt engineering plateaued at 51%, we applied **LoRA (Low-Rank Adaptation)** fine-tuning to adapt Qwen 1.5B to the classification task.

**LoRA Configuration:**
- **Target modules:** `q_proj`, `v_proj` (attention layers)
- Original model weights are **frozen**; only small low-rank adapter matrices are trained
- Libraries: `transformers`, `peft`, `accelerate`, `datasets`

**Training pipeline:**
1. Load AG News and downsample for rapid experimentation
2. Tokenize headlines as prompt-response pairs using a custom `tokenize_for_classification_ft` function
3. Configure training with `TrainingArguments` (batch size, learning rate, epochs)
4. Train with `Trainer`; evaluate using a text generation pipeline
5. Save fine-tuned weights and tokenizer

**Results:**

| Category | F1-Score |
|----------|----------|
| Sports | **0.958** |
| Business | ~0.75 |
| World | 0.705 |
| Sci/Tech | 0.592 |
| **Overall Accuracy** | **~71–73%** |

---

## Results

| Method | Validation Accuracy | Test Accuracy |
|--------|--------------------:|-------------:|
| TF-IDF + Logistic Regression | **92.14%** | **91.67%** |
| Qwen 1.5B — Zero-Shot | — | 47% |
| Qwen 1.5B — Best Prompt (Few-Shot) | — | 51% |
| Qwen 1.5B — LoRA Fine-Tuned | — | ~73% |

> The classical baseline outperforms the best fine-tuned LLM approach by **~20 percentage points**, and the best prompt-based approach by **~41 percentage points**.


> **Note:** `train_split.csv`, `validation.csv`, and subsample files are generated during notebook execution and are not included in the repository due to their size.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `scikit-learn` | TF-IDF vectorization & Logistic Regression |
| `pandas`, `numpy` | Data manipulation |
| `transformers` | Qwen model and tokenizer |
| `peft` | LoRA adapter configuration |
| `accelerate` | Efficient training |
| `datasets` | AG News dataset loading |
| `matplotlib`, `seaborn` | Visualization |

---

*Project completed as part of Generative AI term project.*

*Developed by:*

*-Asude Beyza DEMİRBOĞA*

*-Ayşegül TEKELİ*

*-Ece TİPİCİ*

*-Hatice Müberra GÜL*

