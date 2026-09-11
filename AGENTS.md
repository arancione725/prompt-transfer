# Project Rules

## Scope

This directory is the corrected working copy of the original project. The
original `prompt-transfer-main` directory is read-only input and must not be
modified by work in this copy.

## Structure

- Python package code belongs in `qwen_cross_model/`.
- All projector implementations and projector evaluation helpers belong in
  `qwen_cross_model/projector.py`. Compatibility entrypoints may re-export
  symbols, but must not duplicate the implementation.
- Training outputs belong in a user-specified output directory and should not
  be committed with source code.
- Checkpoints must include the parameters needed to reproduce their prompt,
  especially the SuperPos temperature and seed.

## Validation

- Run `python -m compileall qwen_cross_model` after source changes.
- Run focused helper checks for checkpoint activation, entropy, Top-K
  normalization, and tokenizer ID validation when those paths change.
- Do not use model or dataset downloads as a static validation step.
