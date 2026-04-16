# Change Log - 2026-04-16

## Updated SLM fine-tuning flow

- Standardized RoBERTa initialization in `src/slm/model.py` with a shared helper for inference and fine-tuning.
- Added a backward-compatible `fnetune()` alias that forwards to the new `finetune()` API.
- Aligned fine-tuning defaults with the notebook flow:
  - `weight_decay` changed to `0.01`
  - `epochs` kept at `4` for standalone fine-tuning
  - `batch_size` kept at `32`
  - `warmup_ratio` kept at `0.1`
  - `max_grad_norm` kept at `1.0`
- Updated `finetune_on_clean()` to use class-weighted loss and gradient clipping.
- Added `map_location=self.device` when restoring the best checkpoint to avoid CPU/GPU loading issues.

## Configuration updates

- Updated `src/config.py`:
  - `SLM_FINETUNE_WEIGHT_DECAY = 0.01`

## Verification

- `python -m py_compile src/slm/model.py src/config.py`
- Result: no syntax errors
