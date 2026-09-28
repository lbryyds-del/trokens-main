"""AMP diagnostics: do not silently continue with a collapsed loss scale."""
import math


def checked_amp_scale(scaler, min_scale=0.0):
    """Return scale, failing fast at an explicitly configured safety floor.

    This does not reset a scaler or skip gradients. A tiny scale is a warning
    sign, not proof of underflow on every model; the floor is opt-in per config.
    """
    min_scale = float(min_scale)
    if not math.isfinite(min_scale) or min_scale < 0:
        raise ValueError("SOLVER.AMP_MIN_SCALE must be finite and nonnegative.")
    if not scaler.is_enabled():
        return 1.0
    scale = float(scaler.get_scale())
    if not math.isfinite(scale) or scale <= 0 or scale < min_scale:
        raise RuntimeError(
            f"Unsafe AMP loss scale {scale:g} (SOLVER.AMP_MIN_SCALE={min_scale:g}). "
            "Stopping before silently training with underflowed gradients. "
            "Inspect non-finite gradients and masks. Do not resume the failed "
            "scaler state; for a fresh run use TRAIN.AUTO_RESUME=False and an "
            "empty TRAIN.CHECKPOINT_FILE_PATH in a separate OUTPUT_DIR."
        )
    return scale
