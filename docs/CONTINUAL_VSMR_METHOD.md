# Continual VSMR-Prompt: Canonical Method Specification V1.0

This document is the implementation contract for `--method continual_vsmr`.
It supersedes earlier Continual VSMR drafts. A code change that adds or removes
a mechanism must first revise and review the complete method specification.

## 1. Task and invariants

The task consists of Base session S0 and real incremental sessions S1..ST.
Session t introduces disjoint new classes with K labelled support samples per
class. Evaluation at St covers exactly all classes seen through t.

The method has the following invariants:

- The pretrained ViT-B/16 is frozen for the entire stream.
- A fixed-size shared visual Prompt is used; its component count never grows.
- Base training uses real Base training images.
- Incremental training uses real K-shot support features and historical frozen
  CLS/Patch feature replay. It does not use pseudo-incremental episodes.
- Only the shared Prompt matrix is trainable in incremental sessions.
- Historical component references are immutable after first admission.
- Current local prototypes are recomputed for every seen class after each
  Prompt update.
- Test images and labels never enter training memory or optimization.

## 2. Frozen visual features and memory

For each image, the frozen ViT returns CLS feature
$c_i\in\mathbb R^{768}$ and Patch features
$X_i\in\mathbb R^{196\times768}$ at 224x224 resolution.

The feature memory stores CPU FP16 CLS/Patch features, global class ID,
sample ID and source session. It does not store raw image pixels or test data.
For CUB200, the intended protocol stores five deterministic Base training
exemplars per Base class and all five support features for each incremental
class. The resulting 200 x 5 = 1000 feature records require approximately
288.5 MiB for CLS/Patch features, excluding metadata and Prompt references.

This is frozen-feature replay, not a no-replay protocol.

## 3. Shared Prompt visual decomposition

A fixed Prompt matrix $P\in\mathbb R^{M\times D}$ produces M unordered
components from each frozen Patch feature set.

\[
Q=W_QP,\quad K_i=W_KX_i,\quad V_i=W_VX_i
\]

\[
A_i=\operatorname{softmax}_{N}
\left(\frac{QK_i^\top}{\sqrt D\,\tau}\right)
\]

\[
Z_i=\operatorname{norm}(A_iV_i)
\in\mathbb R^{M\times D}.
\]

The learned projections $W_Q,W_K,W_V$ and global projection $W_g$ are trained
on Base and frozen afterwards. Prompt indices do not represent named semantic
parts; component matching is permutation-invariant.

The global feature is $g_i=\operatorname{norm}(W_gc_i)$. Each class global
prototype is the normalized mean of its admitted global features. Base global
prototypes use all Base training images; each novel global prototype uses its
real K-shot support set. Global prototypes remain fixed after admission.

## 4. OT local prototypes and query reconstruction

For two component sets U,V, define cosine cost
$C_{ab}=1-\langle\operatorname{norm}(u_a),\operatorname{norm}(v_b)\rangle$.
A balanced entropic Sinkhorn coupling $T_\epsilon$ uses uniform marginals
$1/M$. The reported OT distance is the transported cosine cost
$D_\epsilon(U,V)=\langle T_\epsilon,C\rangle$.

Each class local prototype is the OT barycenter of its current Prompt
components. The implementation uses medoid initialization, repeated OT
alignment, normalized barycentric aggregation and a fixed number of outer
iterations.

For query Patch features $X_x$ and class prototype $\Pi_c$, the frozen
query-conditioned reconstructor uses class components as queries and query
Patch features as keys/values:

\[
\widetilde\Pi_c(x)=
\operatorname{norm}\left[
\operatorname{LN}\left(
W_q^r\Pi_c+
\operatorname{Attn}(W_q^r\Pi_c,W_k^rX_x,W_v^rX_x)
\right)\right].
\]

The local score is $s_l(x,c)=-\kappa_l
D_\epsilon(Z_x,\widetilde\Pi_c(x))$.
The reconstructed representation is temporary and is never written into
persistent class memory.

## 5. Global-local classification

\[
s_g(x,c)=\kappa_g\langle g_x,\mu_c^g\rangle
\]

\[
s(x,c)=\alpha s_g(x,c)+(1-\alpha)s_l(x,c).
\]

The scalar fusion coefficient alpha is learned on Base and frozen afterwards.
Prediction is the argmax over all classes seen in the current session.

## 6. Base training

Base uses class-balanced real-image batches with at least two samples per
class, so Leave-One-Out (LOO) prototypes exclude the query itself. Global
prototypes use normalized CLS projections. Local prototypes use OT barycenters.
Training and inference share the same query-conditioned reconstruction and
OT scoring path.

\[
\mathcal L_{base}=\mathcal L_g+
\lambda_l\mathcal L_l+\lambda_j\mathcal L_j
\]

where the three terms are global LOO cross-entropy, local query-conditioned
OT LOO cross-entropy, and fused LOO cross-entropy. No additional attention
overlap, diversity, semantic-alignment or pseudo-incremental objective is
part of V1.0.

After Base training, freeze all model parameters except the shared Prompt.
Select the deterministic Base feature memory, cache CLS/Patch features,
compute Base local prototypes and immutable Base component references.

## 7. Real incremental Prompt adaptation

At Session t, append only the actual K-shot support features for new classes.
The student Prompt starts from the previous Session Prompt. ViT, decomposer
projections, global projection, reconstructor and fusion coefficient remain
frozen. The only trainable parameter is `decomposer.shared_prompts`.

For each class-balanced replay minibatch, re-decompose frozen Patch features
with the current Prompt, construct class OT barycenters, form per-query LOO
candidate prototypes, reconstruct each candidate using the query Patch
features, and compute local and fused all-seen-class logits.

Historical samples are constrained against their immutable first-admission
component references using OT set distance:

\[
\mathcal L_{hist}=
\frac{1}{|\mathcal M_{old}|}
\sum_{i\in\mathcal M_{old}}
D_\epsilon(Z_i^t,Z_i^{ref}).
\]

The incremental objective is:

\[
\mathcal L_{inc}=
\lambda_l\mathcal L_l+
\lambda_j\mathcal L_j+
\lambda_h\mathcal L_{hist}.
\]

No KD, Prompt parameter anchor, previous-Session teacher loss, transfer gate,
base-component retrieval, expert expansion or additional diversity loss is
used in V1.0.

At the end of the Session, recompute all seen-class local prototypes using
the updated Prompt. Keep existing references unchanged. Create permanent
component references only for the newly admitted support samples.

## 8. Training protocol and starting configuration

The CUB200 protocol uses 100 Base classes, 10-way 5-shot incremental
sessions, and 11 evaluated sessions S0..S10. The implementation's initial
configuration uses M=6, OT epsilon 0.05, 30 Sinkhorn iterations, 5 barycenter
iterations, Base learning rate 2e-4, incremental Prompt learning rate 1e-4,
and 20 incremental epochs. These are starting settings, not claimed optimal
values. Hyperparameters must be fixed using Base-only validation or a
predeclared protocol, never final incremental test labels.

## 9. Session-wise evaluation and final metrics

At each session t, evaluate exactly the seen class set
{0,...,base_class+t*way-1}. Every expected class must occur in the evaluation
labels, and every prediction must be within the seen class range. Missing
classes, unexpected labels, out-of-range predictions or skipped sessions
must raise errors rather than be silently clamped or omitted.

For each class c, class accuracy is correct predictions divided by the
number of evaluation samples of c. The Session class-average accuracy is the
unweighted mean over all seen classes:

\[
A_t=\frac{1}{|\bar{\mathcal C}_t|}
\sum_{c\in\bar{\mathcal C}_t}
\operatorname{Acc}_t(c).
\]

Sample-weighted Top-1 accuracy is reported separately and is not used for AA
or PD.

\[
\boxed{AA=\frac{1}{T+1}\sum_{t=0}^{T}A_t}
\]

\[
\boxed{PD=A_0-A_T}
\]

AA is the arithmetic mean of the all-seen-class macro accuracy over S0..ST.
PD is the Base-session macro accuracy minus final-session macro accuracy and
is reported in percentage points. A negative PD is retained if final accuracy
exceeds Base accuracy; it is not clipped to zero.

The experiment writes:

- `session_metrics.csv`: one row per session, with Base, previously seen,
  old novel, current new, all-seen macro and sample Top-1 accuracy.
- `session_metrics.json`: full per-class accuracy for each session.
- `summary.json` and `summary.txt`: AA, PD, definitions and session list.
- `confusion_S{t}.csv`: per-session confusion matrix.
- `metrics.jsonl`: machine-readable training/evaluation event log.
- `continual_vsmr_final.pth`: final model, prototypes, memory and metrics.

## 10. Implementation acceptance checklist

1. ViT parameters never receive gradients and remain in eval mode.
2. Incremental trainable parameter list is exactly
   `decomposer.shared_prompts`.
3. Prompt component count is constant across S0..ST.
4. Base training uses only real Base data and the three specified losses.
5. Incremental sessions use real support features plus frozen historical
   CLS/Patch feature replay.
6. OT barycenter, query reconstruction and OT local score are shared by
   training and inference.
7. LOO excludes the query from its own class prototype.
8. Historical references are immutable; current local prototypes are
   recomputed after each Session.
9. Evaluation covers exactly all seen classes and rejects invalid predictions.
10. AA and PD are computed from the complete ordered S0..ST macro-accuracy
    sequence, not from sample-weighted accuracy.
11. Unit tests, compilation, and a real-data smoke test pass before reporting
    an experiment as verified.
