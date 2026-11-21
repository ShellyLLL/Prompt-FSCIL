# Continual VSMR-Prompt V1.0 — Canonical Implementation Contract

This file is the sole method specification for `--method continual_vsmr`.
It supersedes previous drafts in this repository. If implementation and this
contract disagree, do not run a paper experiment until the discrepancy is
resolved and both code and specification are reviewed.

## 1. FSCIL protocol and invariants

The stream has Base session S0 and incremental sessions S1..ST. Each
incremental session introduces disjoint classes with K labelled support
images per class. Evaluation after session t covers exactly all classes
introduced in S0..St.

- Freeze the pretrained ImageNet-21K ViT-B/16 for the entire stream.
- Use one fixed-size shared visual Prompt with M components. M never expands.
- Base training uses real Base training images; no pseudo-incremental episodes.
- Incremental adaptation uses real K-shot support plus historical frozen CLS
  and Patch feature replay.
- Only `decomposer.shared_prompts` is trainable in incremental sessions.
- Keep each sample's first-admission component reference immutable.
- Recompute all seen-class local prototypes in the current Prompt space after
  every incremental session.
- Test images and labels never enter training memory or optimization.

## 2. Frozen ViT features and feature memory

For each image x, the frozen ViT returns normalized CLS feature
c in R^768 and normalized final Patch features X in R^(196 x 768), at
224 x 224 input resolution.

The memory stores CPU FP16 CLS/Patch features, global class ID, sample ID and
source session. It does not store raw image pixels or test examples. For
CUB200, the intended budget is five deterministic Base training exemplars per
Base class and all five labelled support features for each novel class.
This is frozen-feature replay, not a no-replay method.

The Base-trained global projection W_g produces normalized global features
g = norm(W_g c). Global class prototypes are normalized means of all admitted
global features for each class. Because W_g and CLS features are frozen,
previous global prototypes remain valid and are not recomputed.

## 3. Prompt-conditioned visual decomposition

The Base-trained decomposer contains W_Q, W_K, W_V, W_g and a fixed shared
Prompt matrix P in R^(M x 768). For each image:

```math
Q = W_Q P,   K = W_K X,   V = W_V X
A = softmax_N(Q K^T / (sqrt(768) tau))
Z = norm(A V) in R^(M x 768)
```

The M components form an unordered set. Component indices are not semantic
part labels. W_Q, W_K, W_V and W_g are learned in Base and frozen afterwards.
Only P is updated during incremental sessions.

## 4. OT local prototypes and query-conditioned reconstruction

For component sets U,V, define cosine cost
C_ab = 1 - <norm(u_a), norm(v_b)>. Balanced entropic Sinkhorn transport
T_epsilon uses uniform marginals 1/M. The OT distance is the transported
cosine cost <T_epsilon,C>.

Each seen class local prototype is the OT barycenter of its current Prompt
components. The implementation uses medoid initialization, repeated OT
alignment and normalized barycentric aggregation.

For a query, class prototype components are the reconstructor queries and the
query Patch tokens are keys/values. The residual-plus-attention output is
LayerNorm'ed and normalized. It is a temporary query-conditioned class
representation; it is never written to persistent memory. The local class
score is the negative balanced OT cost between query components and the
reconstructed class components.

Global and local scores are fused by one learned scalar alpha:
s = alpha * s_global + (1-alpha) * s_local. Alpha is learned in Base and
frozen afterwards. Prediction is argmax over all currently seen classes.

## 5. Leave-one-out (LOO) training

A training query must not contribute to its own class prototype. For each
query i, its own-class candidate prototype is recomputed from all admitted
same-class memory samples except i. Other-class candidates use all admitted
samples of those classes. Every class must have at least two memory samples.
Base batches are class-balanced with at least two images per class, allowing
the same LOO principle in Base training.

Training and inference share the same query-conditioned reconstruction and
OT scoring implementation. The Base global LOO logits likewise exclude the
query from its own global prototype.

## 6. Base objective

Base training uses real Base images and three losses only:

```math
L_base = L_global + lambda_local L_local + lambda_joint L_joint
```

L_global is global LOO cross-entropy; L_local is query-conditioned local
OT-LOO cross-entropy; L_joint is cross-entropy on the fused logits. The
decomposer projections, global projection, query reconstructor and fusion
coefficient are learned in Base. The ViT remains frozen.

No attention-overlap, component-diversity, semantic-alignment, distillation,
Prompt-anchor or pseudo-incremental loss is part of V1.0.

After Base training, select a deterministic class-balanced Base feature
memory, cache frozen CLS/Patch features, compute Base local prototypes and
immutable Base component references. Freeze every parameter except the shared
Prompt matrix for subsequent sessions.

## 7. Real incremental Prompt adaptation

At session t, append only the actual K-shot support features for new classes.
Initialize the student Prompt from the previous session Prompt. Keep ViT,
W_Q, W_K, W_V, W_g, query reconstructor and alpha frozen. The only trainable
parameter is `decomposer.shared_prompts`.

For class-balanced replay minibatches, re-decompose the cached Patch features
using the current Prompt, construct class OT barycenters and query-specific
LOO candidate prototypes, reconstruct query-conditioned class components,
and compute local and fused logits over all seen classes.

Historical samples are constrained against their immutable first-admission
component references with permutation-invariant OT distance:

```math
L_hist = mean_{i in historical memory} D_epsilon(Z_i^t, Z_i^ref)
L_inc = lambda_local L_local + lambda_joint L_joint + lambda_hist L_hist
```

Here historical memory means samples admitted before the current session.
New support samples receive their permanent component references only after
the current Prompt optimization is complete. Existing references are never
overwritten. At the end of the session, recompute every seen-class local
prototype using the updated Prompt.

V1.0 excludes previous-Session teacher distillation, KD, Prompt parameter
anchors, base-component retrieval, transfer gates, top-r cross-class
component transfer, expert expansion, Prompt pools, text branches and
additional diversity losses.

## 8. Session-wise evaluation and AA/PD

At each session t, evaluate exactly classes {0,...,base_class+t*way-1}.
Every expected class must occur in the evaluation labels. Every prediction
must lie in the seen-class range. Missing/extra labels, invalid predictions,
or skipped sessions must raise errors; do not silently clamp or omit them.

For each seen class c, class accuracy is correct predictions divided by the
number of evaluation samples of c. The session class-average accuracy is the
unweighted mean across all seen classes:

```math
A_t = (1 / |C_seen,t|) * sum_{c in C_seen,t} Acc_t(c)
AA = (1 / (T+1)) * sum_{t=0}^T A_t
PD = A_0 - A_T
```

AA is the arithmetic mean of all-seen-class macro Top-1 accuracy over S0..ST.
PD is Base-session macro accuracy minus final-session macro accuracy and is
reported in percentage points. A negative PD is valid and must not be clipped.
Sample-weighted Top-1 is reported separately and is never substituted for AA
or PD. A final report must contain the complete expected S0..ST sequence.

The experiment writes:
- `session_metrics.csv`: one row per session, with explicit percentage units
  and separate Base, previously seen, old novel, current new, all-seen macro
  and sample Top-1 accuracy columns.
- `session_metrics.json`: complete per-class accuracy for every session.
- `summary.json` and `summary.txt`: AA, PD, definitions and session list.
- `confusion_S{t}.csv`: per-session confusion matrix.
- `metrics.jsonl`: machine-readable training/evaluation log.
- `continual_vsmr_final.pth`: final model, prototypes, feature memory,
  immutable references and metrics.

## 9. Architecture acceptance

1. ViT parameters never receive gradients and remain in eval mode.
2. Incremental trainable parameters are exactly
   `decomposer.shared_prompts`.
3. Prompt component count remains M in every session.
4. Base uses real Base data and exactly the three specified Base losses.
5. Incremental sessions use real K-shot support plus frozen historical
   CLS/Patch feature replay.
6. OT barycenter, query reconstruction and OT local score are shared by
   training and inference.
7. LOO excludes each query from its own class prototype.
8. Historical references remain immutable; current local prototypes are
   recomputed after every session.
9. Evaluation covers exactly all seen classes and rejects invalid predictions.
10. AA/PD use the complete ordered S0..ST all-seen-class macro-accuracy list.
11. Unit tests, compilation, and real-data smoke tests pass before a full run
    is described as verified.
