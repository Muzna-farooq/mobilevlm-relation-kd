# MobileVLM Relation-Based Knowledge Distillation

Implemented by Muzna Farooq as part of an M2 DKAI Frugal AI group project at Université Paris-Saclay.

## Overview

This experiment trains a MobileVLM V2 3B student to match a frozen MobileVLM V2 7B teacher's text–image relationships on GQA.

Relations are cosine-similarity matrices computed from final decoder hidden states between prompt text tokens and image tokens. Answer and end-of-sequence positions are excluded.

This is a custom relation-matching objective, not an exact reproduction of Park et al.'s RKD or Align-KD.

## Models and Training

- Teacher: mtgv/MobileVLM_V2-7B, all weights frozen.
- Student: mtgv/MobileVLM_V2-3B.
- Vision encoder and projector remain frozen.
- Only language-model LoRA adapters are trained.
- LoRA rank: 16; alpha: 32; dropout: 0.05.
- Loss: gold-answer cross-entropy + λ × relation MSE.
- Relation-loss weight: λ = 1.0.
- Teacher relations are cached on training examples only.
- Training: one epoch, seed 42.
- Learning rate: 0.0002.
- Batch size: 1; gradient accumulation: 16.
- Optimizer updates: 1,635.
- Hardware: Kaggle Tesla T4 ×2.

## Dataset

We use the team's prepared GQA splits.

| Split | Images | Questions |
|---|---:|---:|
| Training | 2,000 | 26,155 |
| Development | 1,000 | 13,174 |
| Testdev | 398 | 12,578 |

## Running the Notebook

1. Import muzna-relation-kd.ipynb into Kaggle.
2. Attach the team's prepared GQA dataset.
3. Enable Internet and select GPU T4 ×2.
4. Install the notebook's pinned dependencies.
5. Restart the kernel after installation.
6. Check dataset paths and output directories.
7. Use the smoke profile for an initial pipeline check.
8. Use the main profile for the full experiment.
9. Run teacher caching, student training, and evaluation in order.
10. Download and preserve the results and adapter files.

Pinned dependencies:

- transformers==4.37.2
- tokenizers==0.15.2
- accelerate==0.27.2
- peft==0.7.1

The notebook requires the prepared dataset and model downloads; it is not a standalone dataset package.

## Results

Accuracy uses normalized exact-match scoring.

| Model | Dev accuracy (%) | Testdev accuracy (%) |
|---|---:|---:|
| Teacher 7B | 71.63 | 62.61 |
| Initial student 3B | 69.34 | 61.03 |
| Relation-KD student 3B | 69.40 | 60.67 |

Teacher and initial-student scores are team reference results. Relation-KD scores come from this experiment.

Compared with the initial student:

- Dev accuracy increased by 0.06 percentage points.
- Testdev accuracy decreased by 0.36 percentage points.
- Dev relation-question accuracy decreased from 62.37% to 61.58%.

## Relation-KD Student: Testdev Efficiency

| Metric | Measured value |
|---|---:|
| Total parameters | 3,024,759,296 |
| Trainable LoRA parameters | 10,485,760 |
| Mean generation latency | 218.49 ms/question |
| Mean throughput | 14.82 tokens/second |
| Peak allocated GPU memory | 6.32 GB |

Timing measures generation only, excluding image preprocessing. LoRA adapters were unmerged.

## Interpretation and Limitations

The training and evaluation pipeline completed successfully, but the added relation objective did not show a clear accuracy benefit in this configuration.

Only one relation-loss weight and one random seed were tested. These results do not establish statistical significance or isolate the effect of relation matching from ordinary supervised fine-tuning.

Next steps include a matched gold-answer-only training control, multiple random seeds, and relation-loss tuning on development data only.

## References

- Chu et al. (2024). MobileVLM V2: Faster and Stronger Baseline for Vision Language Model. https://arxiv.org/abs/2402.03766
- Park et al. (2019). Relational Knowledge Distillation. CVPR. https://arxiv.org/abs/1904.05068
- Hudson and Manning (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. CVPR. https://arxiv.org/abs/1902.09506
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
