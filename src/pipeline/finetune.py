"""
SLM fine-tuning wrapper for MRCD pipeline.
Conditional fine-tuning on D_clean after each round.

Strategy:
- Round < ENABLE_FULL_FINETUNE_FROM_ROUND hoặc D_clean nhỏ: head-only fine-tune (nhanh)
- Round >= ENABLE_FULL_FINETUNE_FROM_ROUND và D_clean >= SLM_FULL_FINETUNE_MIN_SAMPLES:
  full fine-tune (backbone + head) với LR thấp hơn để tránh catastrophic forgetting
"""

from src.config import (
    ENABLE_SLM_FINETUNE,
    SLM_FINETUNE_EPOCHS,
    SLM_FINETUNE_BATCH_SIZE,
    SLM_FINETUNE_LR,
    SLM_FINETUNE_WEIGHT_DECAY,
    SLM_FINETUNE_MIN_SAMPLES,
    ENABLE_FULL_FINETUNE_FROM_ROUND,
    SLM_FULL_FINETUNE_EPOCHS,
    SLM_FULL_FINETUNE_LR,
    SLM_FULL_FINETUNE_MIN_SAMPLES,
)


def maybe_finetune_slm_on_clean(
    slm,
    clean_pool: list,
    round_id: int,
    enable_slm_finetune: bool = ENABLE_SLM_FINETUNE,
    slm_finetune_epochs: int = SLM_FINETUNE_EPOCHS,
    slm_finetune_batch_size: int = SLM_FINETUNE_BATCH_SIZE,
    slm_finetune_lr: float = SLM_FINETUNE_LR,
    slm_finetune_weight_decay: float = SLM_FINETUNE_WEIGHT_DECAY,
    slm_finetune_min_samples: int = SLM_FINETUNE_MIN_SAMPLES,
) -> dict:
    """
    Conditionally fine-tune SLM on D_clean.

    Strategy:
    - Nếu round >= ENABLE_FULL_FINETUNE_FROM_ROUND VÀ D_clean >= SLM_FULL_FINETUNE_MIN_SAMPLES:
      Dùng FULL fine-tune (backbone + head) với LR thấp (2e-5) — giữ ngữ nghĩa PhoBERT,
      đồng thời cập nhật representation để phân biệt fake/real tốt hơn.
    - Ngược lại: Dùng head-only fine-tune (nhanh hơn, an toàn khi ít data).

    Skips if:
    - Fine-tuning is disabled
    - Not enough clean samples (< min_samples)

    Args:
        slm: IntegratedSLM instance to fine-tune
        clean_pool: List of clean sample dicts
        round_id: Current round number (for logging and strategy decision)

    Returns:
        dict with training statistics and strategy used
    """
    if not enable_slm_finetune:
        return {"trained": False, "reason": "disabled"}
    if len(clean_pool) < slm_finetune_min_samples:
        return {
            "trained": False,
            "reason": "insufficient_samples",
            "available_samples": len(clean_pool),
            "min_samples": slm_finetune_min_samples,
        }

    # === Xác định strategy ===
    use_full_finetune = (
        round_id >= ENABLE_FULL_FINETUNE_FROM_ROUND
        and len(clean_pool) >= SLM_FULL_FINETUNE_MIN_SAMPLES
    )

    if use_full_finetune:
        # Full fine-tune: cập nhật cả backbone + head
        print(
            f"[Round {round_id}] Full fine-tune SLM (backbone + head) trên "
            f"{len(clean_pool)} D_clean samples "
            f"(LR={SLM_FULL_FINETUNE_LR}, epochs={SLM_FULL_FINETUNE_EPOCHS})..."
        )
        # Lấy texts và labels từ clean_pool
        train_texts = [s["text"] for s in clean_pool if s.get("text")]
        train_labels = [
            int(s.get("label", s.get("label_slm", 1)))
            for s in clean_pool
            if s.get("text")
        ]
        stats = slm.finetune_full(
            train_texts=train_texts,
            train_labels=train_labels,
            epochs=SLM_FULL_FINETUNE_EPOCHS,
            batch_size=slm_finetune_batch_size,
            lr=SLM_FULL_FINETUNE_LR,
            weight_decay=slm_finetune_weight_decay,
            warmup_ratio=0.1,
        )
        stats["strategy"] = "full_finetune"
    else:
        # Head-only fine-tune: nhanh hơn, an toàn khi ít data
        print(
            f"[Round {round_id}] Head-only fine-tune SLM trên "
            f"{len(clean_pool)} D_clean samples "
            f"(LR={slm_finetune_lr}, epochs={slm_finetune_epochs})..."
        )
        stats = slm.finetune_on_clean(
            clean_samples=clean_pool,
            epochs=slm_finetune_epochs,
            batch_size=slm_finetune_batch_size,
            lr=slm_finetune_lr,
            weight_decay=slm_finetune_weight_decay,
        )
        stats["strategy"] = "head_only_finetune"

    if stats.get("trained", False):
        strategy = stats.get("strategy", "unknown")
        avg_loss = stats.get("avg_loss", stats.get("train_loss_history", [None])[-1])
        avg_loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
        print(
            f"SLM fine-tune done | round={round_id} strategy={strategy} "
            f"samples={stats.get('samples', len(clean_pool))} "
            f"epochs={stats.get('epochs', '?')} "
            f"avg_loss={avg_loss_str}"
        )
    else:
        print(f"Skip SLM fine-tune at round {round_id}: {stats}")

    return stats
