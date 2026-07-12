# TEAL MLU full-data results (epoch 10, seed 42)

This directory contains the reviewable outputs from the three full-data TEAL MLU runs. Lower normalized MLU is better; `1.0` matches the per-sample optimum.

The runs use upstream `mlu` commit `0229a19801919d4dccc5a60f3b46fe0a27d7f08b` as their source base and the corrected training logic in this pull request. Changes made after the runs only improve portability, compatibility handling, tests, and artifact sanitization; they do not change the training math used for these results.

## Common setting

| Setting | Value |
| --- | ---: |
| Objective | `mlu` |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Seed | 42 |
| Paths per source-destination pair | 4 |
| FlowGNN layers | 6 |
| COMA samples | 5 |
| ADMM steps | 0 |
| Checkpoint load / save | false / false |
| Device used | `cuda:0` |

## Data splits

| Dataset | Train | Validation | Test | Split rule |
| --- | ---: | ---: | ---: | --- |
| DynGEANT | 5,000 | 1,000 | 4,000 | 50 clusters of 200 samples: clusters 0-24 / 25-29 / 30-49 |
| geant | 6,463 | 1,616 | 2,693 | First 75% is seeded and split 80/20 for train/validation; final 25% is test |
| abilene | 28,857 | 7,215 | 12,024 | First 75% is seeded and split 80/20 for train/validation; final 25% is test |

## Test results

These statistics are computed from `result.json` -> `results`, i.e. predicted MLU divided by optimal MLU.

| Dataset | Samples | Mean | P50 | P90 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DynGEANT | 4,000 | 5.574393 | 5.419145 | 9.828781 | 11.901934 | 13.571401 |
| geant | 2,693 | 1.574350 | 1.522329 | 1.820545 | 2.196774 | 2.750560 |
| abilene | 12,024 | 1.212629 | 1.205658 | 1.325182 | 1.427552 | 1.613421 |

## Reproduction

Point `DATA_BASE` at a directory containing the MLU-labelled `DynGEANT`, `geant`, and `abilene` subdirectories, then run:

```bash
DATA_BASE=/path/to/mlu-data \
DATASETS="DynGEANT geant abilene" \
EPOCHS=10 BATCH_SIZE=16 LEARNING_RATE=0.001 SEED=42 \
MODEL_LOAD=false MODEL_SAVE=false \
bash teal_mlu_run.sh
```

## Files

Each dataset directory contains:

- `result.json`: the complete machine-readable output. `results` stores normalized MLU, `raw_mlu` stores predicted MLU before normalization, and `settings` records the run configuration. Machine-local paths were normalized to `<DATA_ROOT>`, `<MODEL_ROOT>`, and `<RESULT_ROOT>`; `checkpoint_path` was reconstructed from the model path recorded in each log.
- `run.txt`: a sanitized review extract from the original terminal log. Repeated tqdm carriage-return frames were collapsed to the final train and validation state for every epoch, followed by the final test state and statistics.

The retained PyTorch pickle-protocol warning comes from loading dataset tensors, not from loading a model checkpoint; all three runs used `model_load=false`.
