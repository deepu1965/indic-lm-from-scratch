# Phase 3 Report

## 1. Synthetic Reasoning Dataset

The synthetic logical reasoning dataset was generated for both Hindi and Nepali.

| Metric | Value |
|---|---|
| Total Examples (per language) | 7,000 |
| Train Split | 5,600 |
| Validation Split | 700 |
| Test Split | 700 |

### Task Families
| Family | Count |
|---|---|
| direct_comparison | 1,000 |
| largest_smallest | 1,000 |
| general_comparison | 1,000 |
| two_hop_transitivity | 1,000 |
| three_hop_transitivity | 1,000 |
| reversed_relation | 1,000 |
| mixed_equality | 1,000 |

### Distribution
* **Attributes:** height (1750), price (1750), age (1750), quantity (1750)
* **Difficulty:** easy (2100), medium (2800), hard (2100)
* **Representation:** relational (5250), numerical (1750)

### Leakage Controls
* **Entity Name Strategy:** Disjoint entity pools were used for each split to prevent train-test leakage. Entity names used in the test set were completely held out from the train and validation sets.
* **Templates:** There are 28 Hindi templates and 28 Nepali templates, making 56 templates in total (4 template variations per task family).

## 2. SFT Setup

The fine-tuning was performed using LoRA (Low-Rank Adaptation) on the frozen Phase 2 base models.

| Hyperparameter | Value |
|---|---|
| Base Model | DecoderOnlyTransformer (12 layers, 6 heads, d_model=384) |
| Tokenizer | BPE v10000 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | `W_q`, `W_k`, `W_v`, `W_o` |
| Trainable Parameters | 294,912 (1.16%) |
| Epochs | 5 |
| Batch Size (micro) | 2 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Scheduler | Cosine with Warmup (warmup ratio = 0.05) |
| Precision | FP16 (AMP) |
| Early Stopping | Patience = 2, Min Delta = 0.001 |
| Loss Masking | Loss computed only on target/answer tokens |

## 3. Pretrained vs Fine-tuned Results

The models were evaluated on the 700 test examples for each language.

### Overall Results

| Language | Model | Reasoning Accuracy | Exact Match |
|---|---|---|---|
| Hindi | Pretrained | 0.29% | 0.00% |
| Hindi | SFT | 42.00% | 39.86% |
| Nepali | Pretrained | 0.00% | 0.00% |
| Nepali | SFT | 14.29% | 10.29% |

After SFT, the Hindi model reached 42.0% reasoning accuracy. The Nepali model also improved, reaching 14.3%.

### Results by Difficulty

| Difficulty | HI Pretrained (Acc / EM) | HI SFT (Acc / EM) | NE Pretrained (Acc / EM) | NE SFT (Acc / EM) |
|---|---|---|---|---|
| Easy | 0.5% / 0.0% | 38.6% / 35.7% | 0.0% / 0.0% | 16.7% / 12.9% |
| Medium | 0.0% / 0.0% | 41.1% / 39.6% | 0.0% / 0.0% | 11.4% / 6.8% |
| Hard | 0.5% / 0.0% | 46.7% / 44.3% | 0.0% / 0.0% | 15.7% / 12.4% |

Here, **Acc** - reasoning accuracy ; **EM** - exact match

### Results by Task Family — Reasoning Accuracy

| Task Family | HI Pretrained | HI SFT | NE Pretrained | NE SFT |
|---|---|---|---|---|
| direct_comparison | 0% | 44% | 0% | 14% |
| largest_smallest | 0% | 19% | 0% | 7% |
| general_comparison | 0% | 39% | 0% | 18% |
| two_hop_transitivity | 0% | 40% | 0% | 9% |
| three_hop_transitivity | 0% | 65% | 0% | 22% |
| reversed_relation | 1% | 46% | 0% | 17% |
| mixed_equality | 1% | 41% | 0% | 13% |

### Results by Task Family — Exact Match

| Task Family | HI Pretrained | HI SFT | NE Pretrained | NE SFT |
|---|---|---|---|---|
| direct_comparison | 0% | 44% | 0% | 10% |
| largest_smallest | 0% | 12% | 0% | 1% |
| general_comparison | 0% | 39% | 0% | 13% |
| two_hop_transitivity | 0% | 38% | 0% | 7% |
| three_hop_transitivity | 0% | 62% | 0% | 18% |
| reversed_relation | 0% | 45% | 0% | 12% |
| mixed_equality | 0% | 39% | 0% | 11% |

## 4. Error Analysis

| Error Category | Hindi Pretrained | Hindi SFT | Nepali Pretrained | Nepali SFT |
|---|---|---|---|---|
| no_recognizable_answer | 620 | 248 | 627 | 502 |
| wrong_entity | 2 | 158 | 0 | 90 |
| equality_relation_missing | 75 | 0 | 73 | 7 |
| ambiguous_multiple_entities | 1 | 0 | 0 | 1 |

### Exact Match vs Reasoning Correctness
* **Hindi SFT:** 15 examples had correct reasoning but failed exact match.
* **Nepali SFT:** 41 examples had correct reasoning but failed exact match.

## 5. Model H vs Model L

Comparing the Hindi model (H) and Nepali model (L) after fine-tuning.

| Metric | Hindi SFT | Nepali SFT |
|---|---|---|
| Overall Reasoning Accuracy | 42.00% | 14.29% |
| Overall Exact Match | 39.86% | 10.29% |

## 6. Qualitative Examples

Here are some actual examples from the evaluation outputs on the comparative-reasoning test set.

**Example 1: Improvement after SFT (Hindi - Direct Comparison)**
* **Input:** दी गई जानकारी के अनुसार, साक्षी की मात्रा स्वाति से अधिक है, स्वाति की मात्रा निधि से अधिक है। साक्षी और स्वाति की तुलना करने पर किसकी मात्रा अधिक है?
* **Expected Answer:** E02 (साक्षी)
* **Pretrained Output:** स्वाति की मात्रा अधिक है। (Incorrect)
* **SFT Output:** साक्षी की मात्रा अधिक है। (Correct)

**Example 2: Equality Case (Nepali - Mixed Equality)**
* **Input:** पार्वतीको मूल्य विवेक भन्दा बढी छ। विवेक र मञ्जुको मूल्य बराबर छ। पार्वतीको मूल्य सरस्वती भन्दा कम छ। यी मध्ये कसको मूल्य कम छ: सरस्वती वा मञ्जु?
* **Expected Answer:** E04 (मञ्जु)
* **SFT Output:** मञ्जुको मूल्य कम छ। (Correct)

## 7. Attention Analysis

The attention analysis was run on all 700 synthetic test examples. 

### Hindi Attention Entropy and Distance

#### Layer 0

| Head | Pre Entropy | SFT Entropy | Pre Distance | SFT Distance |
|---|---|---|---|---|
| H0 | 3.4727 | 3.4108 | 8.3716 | 8.4846 |
| H1 | 3.2052 | 3.1453 | 7.8679 | 7.3373 |
| H2 | 3.4979 | 3.2724 | 8.7995 | 8.3691 |
| H3 | 3.4402 | 3.3435 | 9.0626 | 8.7535 |
| H4 | 3.3950 | 3.3645 | 8.9488 | 8.7112 |
| H5 | 3.5534 | 3.4322 | 11.7167 | 10.8317 |

#### Layer 11

| Head | Pre Entropy | SFT Entropy | Pre Distance | SFT Distance |
|---|---|---|---|---|
| H0 | 1.8935 | 1.5015 | 2.9511 | 8.3107 |
| H1 | 3.1125 | 3.0145 | 9.1014 | 9.4542 |
| H2 | 1.9179 | 1.6894 | 2.8035 | 5.5429 |
| H3 | 2.5482 | 1.0486 | 5.1020 | 14.8146 |
| H4 | 1.8212 | 2.0748 | 2.3081 | 5.3570 |
| H5 | 3.0555 | 2.1787 | 5.7529 | 3.1653 |

### Nepali Attention Entropy and Distance

#### Layer 0

| Head | Pre Entropy | SFT Entropy | Pre Distance | SFT Distance |
|---|---|---|---|---|
| H0 | 2.9991 | 2.8738 | 6.2148 | 5.9983 |
| H1 | 3.3664 | 3.1209 | 6.7636 | 5.9720 |
| H2 | 2.5005 | 2.2295 | 4.5264 | 4.4590 |
| H3 | 1.6330 | 1.5058 | 2.7061 | 2.6942 |
| H4 | 2.8769 | 2.6159 | 5.1196 | 5.0176 |
| H5 | 2.7066 | 2.5574 | 4.7337 | 4.3846 |

#### Layer 11

| Head | Pre Entropy | SFT Entropy | Pre Distance | SFT Distance |
|---|---|---|---|---|
| H0 | 2.8870 | 2.3819 | 7.3443 | 7.8481 |
| H1 | 3.0266 | 2.3799 | 10.7292 | 10.5738 |
| H2 | 3.2389 | 2.6183 | 6.2524 | 4.2084 |
| H3 | 2.2601 | 1.5796 | 9.1699 | 9.9751 |
| H4 | 3.0499 | 1.6358 | 10.3310 | 10.8821 |
| H5 | 3.2833 | 2.3747 | 8.6751 | 8.8951 |


### Selected Head Identities

| Language | Layer | Role | Pretrained Head | SFT Head |
|---|---|---|---|---|
| Hindi | 0 (Early) | Local | H1 | H1 |
| Hindi | 0 (Early) | Long-range | H5 | H5 |
| Hindi | 0 (Early) | Focused | H4 | H2 |
| Hindi | 11 (Late) | Local | H4 | H5 |
| Hindi | 11 (Late) | Long-range | H1 | H3 |
| Hindi | 11 (Late) | Focused | H0 | H0 |
| Nepali | 0 (Early) | Local | H3 | H3 |
| Nepali | 0 (Early) | Long-range | H1 | H0 |
| Nepali | 0 (Early) | Focused | H2 | H2 |
| Nepali | 11 (Late) | Local | H2 | H2 |
| Nepali | 11 (Late) | Long-range | H1 | H4 |
| Nepali | 11 (Late) | Focused | H3 | H3 |


SFT changed the attention patterns of some heads. In particular, the heads selected as local, long-range, and focused based on the measured distance and entropy changed in some layers.

For example, at Hindi Layer 11, the long-range role changed from H1 in the pretrained model to H3 after SFT. These changes in selected local, long-range, and focused heads show changes in attention behavior after SFT.



## 8. Attention Heatmaps

The following heatmaps show attention token-to-token on the comparative reasoning prompt for ID 003010. 
Because the same test example is used for the pretrained vs SFT attention heatmap comparison, the input token sequence is identical.

### Comparative-Reasoning Prompt — Test ID 003010

#### Hindi
**Facts:** दी गई जानकारी यह है कि पंकज की उम्र सिमरन से अधिक है, सिमरन की उम्र प्रवीण से अधिक है।

**Question:** पंकज और प्रवीण की तुलना करने पर किसकी उम्र कम है?

**Expected answer:** E03 (प्रवीण)

#### Nepali
**Facts:** दुर्गाको उमेर विवेक भन्दा बढी छ। विवेकको उमेर सुरेश भन्दा बढी छ।

**Question:** दुर्गा र सुरेश मध्ये कसको उमेर बढी छ?

**Expected answer:** E01 (दुर्गा)

### Hindi — Layer 0 (Early)

| Model | H1 | H2 | H4 | H5 |
|---|---|---|---|---|
| Pretrained | ![Hindi Pre L0 H1](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h1_pretrained.png) | ![Hindi Pre L0 H2](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h2_pretrained.png) | ![Hindi Pre L0 H4](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h4_pretrained.png) | ![Hindi Pre L0 H5](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h5_pretrained.png) |
| SFT | ![Hindi SFT L0 H1](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h1_sft.png) | ![Hindi SFT L0 H2](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h2_sft.png) | ![Hindi SFT L0 H4](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h4_sft.png) | ![Hindi SFT L0 H5](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l0_h5_sft.png) |

**Note:** For Hindi Layer 0:
* Pretrained: Local H1, Long-range H5, Focused H4.
* SFT: Local H1, Long-range H5, Focused H2.

### Hindi — Layer 11 (Late)

| Model | H0 | H1 | H3 | H4 | H5 |
|---|---|---|---|---|---|
| Pretrained | ![Hindi Pre L11 H0](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h0_pretrained.png) | ![Hindi Pre L11 H1](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h1_pretrained.png) | ![Hindi Pre L11 H3](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h3_pretrained.png) | ![Hindi Pre L11 H4](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h4_pretrained.png) | ![Hindi Pre L11 H5](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h5_pretrained.png) |
| SFT | ![Hindi SFT L11 H0](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h0_sft.png) | ![Hindi SFT L11 H1](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h1_sft.png) | ![Hindi SFT L11 H3](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h3_sft.png) | ![Hindi SFT L11 H4](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h4_sft.png) | ![Hindi SFT L11 H5](../phase3/outputs/evaluation/attention/hi/comparison/heatmaps/heatmap_l11_h5_sft.png) |

**Note:** For Hindi Layer 11:
* Pretrained: Local H4, Long-range H1, Focused H0.
* SFT: Local H5, Long-range H3, Focused H0.

### Nepali — Layer 0 (Early)

| Model | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| Pretrained | ![Nepali Pre L0 H0](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h0_pretrained.png) | ![Nepali Pre L0 H1](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h1_pretrained.png) | ![Nepali Pre L0 H2](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h2_pretrained.png) | ![Nepali Pre L0 H3](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h3_pretrained.png) |
| SFT | ![Nepali SFT L0 H0](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h0_sft.png) | ![Nepali SFT L0 H1](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h1_sft.png) | ![Nepali SFT L0 H2](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h2_sft.png) | ![Nepali SFT L0 H3](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l0_h3_sft.png) |

**Note:** For Nepali Layer 0:
* Pretrained: Local H3, Long-range H1, Focused H2.
* SFT: Local H3, Long-range H0, Focused H2.

### Nepali — Layer 11 (Late)

| Model | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| Pretrained | ![Nepali Pre L11 H1](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h1_pretrained.png) | ![Nepali Pre L11 H2](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h2_pretrained.png) | ![Nepali Pre L11 H3](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h3_pretrained.png) | ![Nepali Pre L11 H4](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h4_pretrained.png) |
| SFT | ![Nepali SFT L11 H1](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h1_sft.png) | ![Nepali SFT L11 H2](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h2_sft.png) | ![Nepali SFT L11 H3](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h3_sft.png) | ![Nepali SFT L11 H4](../phase3/outputs/evaluation/attention/ne/comparison/heatmaps/heatmap_l11_h4_sft.png) |

**Note:** For Nepali Layer 11:
* Pretrained: Local H2, Long-range H1, Focused H3.
* SFT: Local H2, Long-range H4, Focused H3.

