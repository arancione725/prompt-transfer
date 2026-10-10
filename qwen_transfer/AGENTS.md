# Project Rules

## Scope

This project contains the complete Qwen2.5-1.5B to Qwen2.5-7B SuperPos
workflow: source training, direct transfer initialization, label-free RFA
gradient refinement, and isolated target evaluation. Do not import or mutate
the previous `prompt-transfer-main*` workspaces at runtime.

## Structure

- Package code belongs in `qwen_cross_model/`.
- Shared RFA mathematics belongs in `qwen_cross_model/rfa.py`.
- SuperPos parameterization and checkpoint logic belong in
  `qwen_cross_model/superpos.py`.
- Source training belongs in `qwen_cross_model/train_superpos.py`.
- Label-free target optimization belongs in `qwen_cross_model/refine_target.py`.
- Label-reading target evaluation belongs in `qwen_cross_model/evaluate.py` and
  must remain separate from refinement.
- The end-to-end entrypoint belongs in `qwen_cross_model/pipeline.py`.
- Multi-K cache preparation, fixed-support refinement orchestration, and
  label-free K selection belong in `qwen_cross_model/sweep_k.py`.
- Post-hoc labeled evaluation of a completed K sweep belongs in
  `qwen_cross_model/evaluate_sweep.py`; sweep preparation and selection must
  never import this module or labeled dataset helpers.
- Unit tests use `qwen_cross_model/test_*.py` and must not download models or
  datasets.
- Generated reports belong in a user-specified output directory and must not
  overwrite existing files.

## Method Constraints

- Refinement must not read target labels, accuracies, or target-task metrics.
- Source and target model parameters must stay frozen during refinement; only
  the target SuperPos coordinate logits may receive gradients.
- The target coordinates start from the saved source coordinates.
- A basis-K run uses a fixed per-position source Top-K support throughout
  refinement and compares against the same full source functional signature.
- K selection uses disjoint unlabeled train/validation probe banks and the
  one-standard-error rule; accuracy may not influence K selection.
- Token IDs must be fully aligned across tokenizers before comparing logits.
- Full aligned non-special vocabulary is the primary score.
- Sparse anchors must be derived only from source-model behavior, fixed before
  target optimization, and include an OTHER mass bucket.
- The refinement objective is RFA plus a KL stay term to the source coordinate
  distribution. It is label-free/zero-shot refinement, not training-free.
- A checkpoint must record enough metadata to reconstruct its prompt,
  especially seed, weight mode, and SuperPos temperature.

## Validation

- Run `python -m unittest discover -s qwen_cross_model -p "test_*.py" -v`.
- Run `python -m compileall qwen_cross_model`.
- Run CLI `--help` smoke checks for training, refinement, single-checkpoint
  evaluation, the pipeline, K sweep, and sweep evaluation after argument
  changes.
- Do not download model or dataset assets as part of static validation.
