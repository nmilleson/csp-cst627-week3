# CivicDesk 311 Ticket Rewriter — Training & Evaluation Summary

## Hardware

Free-tier Colab GPU used for both training runs:

- **GPU:** Tesla T4 — 15,360 MiB (≈16 GB VRAM)
- **Driver:** 580.82.07 — **CUDA:** 13.0

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   45C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## 1. SFT (LoRA) Training

3 epochs, 51 steps total (270 training examples, effective batch size 16).

- **Wall-clock:** 286.8s
- **Peak GPU memory:** 1.21 GB
- **Final train loss:** 1.06

Eval loss per epoch — steady improvement:

| epoch | eval_loss | eval_mean_token_accuracy |
|---|---|---|
| 1 | 0.8794 | 0.8201 |
| 2 | 0.4416 | 0.9061 |
| 3 | 0.4064 | 0.9125 |

Final summary line: `train_runtime=286.2s, train_samples_per_second=2.83, train_loss=1.06, epoch=3`

## 2. DPO Training (on top of the SFT adapter)

1 epoch, 34 steps total (270 preference pairs, effective batch size 8).

- **Wall-clock:** 241.3s
- **Peak GPU memory:** 5.87 GB
- **Final train loss:** 0.472

Reward metrics over the course of training — the policy learns to separate chosen from rejected tickets:

| step (epoch) | rewards/accuracies | rewards/margins |
|---|---|---|
| 0.30 | 0.650 | 0.070 |
| 0.59 | 0.975 | 0.886 |
| 0.89 | 0.900 | 1.202 |
| 1.0 (eval) | 0.969 | 1.353 |

Final summary line: `train_runtime=240.9s, train_samples_per_second=1.121, train_loss=0.472, epoch=1`

## 3. Evaluation: Base vs. SFT vs. DPO

Same 30 held-out complaints, greedy decoding, scored against the ground-truth ticket.

| metric | base | sft | dpo |
|---|---|---|---|
| valid_json_rate | 1.000 | 1.000 | 1.000 |
| has_all_fields_rate | 1.000 | 1.000 | 1.000 |
| category_accuracy | 0.000 | 1.000 | 0.933 |
| priority_accuracy | 0.067 | 0.600 | 0.600 |
| exact_match_rate | 0.000 | 0.033 | 0.000 |
| seconds_per_example | 3.780 | 5.143 | 5.027 |

SFT is the biggest jump (category accuracy 0.0 → 1.0). DPO's category accuracy and exact-match rate are slightly lower than SFT alone on this 30-example val set — worth a closer look at which examples regressed before presenting this as a strict improvement.

### Raw per-run summaries

**Base model (no adapter):**
```json
{
  "label": "base",
  "n_examples": 30,
  "wall_clock_seconds": 113.40273308753967,
  "seconds_per_example": 3.780091102917989,
  "valid_json_rate": 1.0,
  "has_all_fields_rate": 1.0,
  "category_accuracy": 0.0,
  "priority_accuracy": 0.06666666666666667,
  "exact_match_rate": 0.0
}
```

**SFT adapter:**
```json
{
  "label": "sft",
  "n_examples": 30,
  "wall_clock_seconds": 154.29626655578613,
  "seconds_per_example": 5.1432088851928714,
  "valid_json_rate": 1.0,
  "has_all_fields_rate": 1.0,
  "category_accuracy": 1.0,
  "priority_accuracy": 0.6,
  "exact_match_rate": 0.03333333333333333
}
```

**DPO adapter:**
```json
{
  "label": "dpo",
  "n_examples": 30,
  "wall_clock_seconds": 150.81807208061218,
  "seconds_per_example": 5.027269069353739,
  "valid_json_rate": 1.0,
  "has_all_fields_rate": 1.0,
  "category_accuracy": 0.9333333333333333,
  "priority_accuracy": 0.6,
  "exact_match_rate": 0.0
}
```

## 4. Hardware Budget Conclusion

Free-tier Colab T4 provides 16 GB VRAM. Peak usage across both training runs — **1.21 GB for SFT**, **5.87 GB for DPO** — comfortably fits within that budget, with substantial headroom left over (e.g. for a larger batch size or longer sequences).
