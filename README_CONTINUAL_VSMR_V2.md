# Continual VSMR-Prompt V2.0

This document describes the `--method continual_vsmr` implementation on
`feature/continual-vsmr-peft`. It is separate from the legacy `--method vsmr`
implementation and does not change its OT-based method.

## Architecture

- Frozen pretrained ImageNet-21K ViT-B/16.
- One shared external visual Prompt.
- Base-trained Q/K/V projections for Prompt-guided Patch context aggregation.
- A frozen CLS global projection and a learned convex Global/Local fusion weight.
- Normalized mean prototypes for both global and local representations.
- Incremental Prompt-only updates with two immutable historical anchors:
  sample-level Birth Reference and class-level Birth Prototype.
- Exactly five stored frozen-feature samples per seen class (Base exemplars and
  all K-shot incremental supports). This is a feature-replay protocol, not a
  no-replay protocol.

The incremental optimizer updates only `context.shared_prompt`. The ViT,
Q/K/V, global projection, and fusion coefficient remain frozen.

## Objectives

Base training uses balanced batches and exact leave-one-out (LOO) cosine
prototypes for each query:

`L_base = w_g L_global + w_l L_local + w_j L_joint`

Incremental training compares each new support sample with every seen class,
using LOO for its own class. The objective is:

`L_inc = L_new + w_rep L_rep + w_proto L_proto`

- `L_rep`: mean cosine drift of historical memory samples from their fixed
  class-introduction representations.
- `L_proto`: mean cosine drift of historical class local prototypes from
  their fixed class-introduction prototypes.

At the end of each session, all seen-class local prototypes are rebuilt in
the current Prompt representation space. Historical Birth References are
never overwritten; only newly admitted samples/classes receive references.

## Runtime and memory design

The training path is deliberately feature-cached:

1. Extract Base train CLS/Patch features once with deterministic ViT-consistent
   preprocessing. Cache them on CPU in FP16.
2. Train Prompt/Q/K/V/global projection/fusion on cached features. No ViT
   forward is repeated across Base epochs.
3. After Base, retain five examples per Base class and cache their CLS plus
   projected Patch K/V in CPU FP16.
4. For each incremental session, run ViT only on the new K-shot supports.
   Store their CLS and projected K/V in the same CPU FP16 memory.
5. Each Prompt update recomputes context from cached K/V, in chunks controlled
   by `--continual_feature_batch_size`. It never reruns the ViT.
6. Build the complete held-out test feature cache once after Base. All S0..ST
   evaluations reuse cached CLS/K/V and only recompute current Prompt context
   and prototype scores.

The complete CUB test cache is stored in CPU FP16. Approximate storage for
5794 images is about 3.5 GB for projected K and V, plus CLS. Ensure the host
has sufficient RAM. Reduce `--continual_feature_batch_size` if GPU memory is
limited; it does not change the mathematical objective.

## CUB-200 run

```bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

python run_script/vsmr_cub200.py
```

Equivalent direct launch:

```bash
python train.py --method continual_vsmr -project base -dataset cub200 -vit \
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \
  -gpu 0 -seed 1 -epochs_base 50 -epochs_new 10 -num_workers 4 \
  -test_batch_size 64 --continual_base_memory_per_class 5 \
  --continual_feature_batch_size 128 --continual_eval_batch_size 128
```

## Main options

| Option | Default | Meaning |
|---|---:|---|
| `continual_base_memory_per_class` | 5 | Fixed Base memory budget; V2 requires exactly 5 |
| `continual_prompt_lr` | 1e-4 | Incremental Prompt learning rate |
| `continual_history_rep_weight` | 0.1 | Historical sample representation consistency |
| `continual_history_proto_weight` | 0.1 | Historical class prototype consistency |
| `continual_global_init` | 0.5 | Initial Base fusion weight |
| `continual_global_loss_weight` | 1.0 | Base global classification loss |
| `continual_local_loss_weight` | 1.0 | Base local classification loss |
| `continual_joint_loss_weight` | 1.0 | Base fused classification loss |
| `continual_classes_per_batch` | 16 | Base balanced classes per batch |
| `continual_samples_per_class` | 4 | Base balanced samples per class |
| `continual_feature_batch_size` | 128 | Cached feature chunk size |
| `continual_eval_batch_size` | 128 | Cached evaluation chunk size |

## Outputs and acceptance

Each run writes config, per-epoch JSONL logs, per-session confusion matrices,
session metrics, summary, and Base/per-session/final checkpoints under
`checkpoint/<out>/<dataset>/ContinualVSMR_V2/<timestamp>_seed<seed>/`.

The summary reports:
- AA: arithmetic mean of all-seen-class macro Top-1 accuracy over S0..ST.
- PD: S0 all-seen-class macro accuracy minus ST all-seen-class macro accuracy.
- Base-class accuracy drop as a separate diagnostic.
- Feature extraction, Base training, incremental session, and total runtime.

Before a real-data run, execute:

```bash
python -m py_compile train.py models/base/continual_vsmr.py models/base/continual_vsmr_model.py models/base/continual_vsmr_trainer.py
python -m unittest discover -s tests -p 'test_*vsmr*.py' -v
```

A successful synthetic test is an architecture check, not evidence of AA/PD
improvement. CUB-200 S0-S10 is the next real-data acceptance stage.
