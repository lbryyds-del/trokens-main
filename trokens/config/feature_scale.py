"""Opt-in, fixed gains at the DinoTxt -> Trokens feature interface."""
import math


def validate_feature_scale_cfg(cfg):
    scale = cfg.MODEL.FEATURE_SCALE
    for key in ("APPEARANCE_GAIN", "HOD_GAIN", "CTD_GAIN"):
        value = float(scale[key])
        if not math.isfinite(value) or value < 0 or (key == "APPEARANCE_GAIN" and value == 0):
            raise ValueError(f"FEATURE_SCALE.{key} must be finite and "
                             + ("positive." if key == "APPEARANCE_GAIN" else "nonnegative."))
    if not scale.ENABLE:
        return
    if cfg.MODEL.FEAT_EXTRACTOR != "dinotxt_vitl14_reg4":
        raise ValueError("FEATURE_SCALE currently supports only dinotxt_vitl14_reg4.")
    if cfg.MODEL.TRAIN_BACKBONE or cfg.MODEL.APPEARANCE_MODULE_DISABLE:
        raise ValueError("FEATURE_SCALE requires the frozen, enabled DinoTxt visual tower.")
    # This checkout only implements early addition. Do not import the remote
    # separate-motion experiment merely to share its scale adapter.
    if getattr(cfg.MODEL.MOTION_MODULE, "FUSION_MODE", "early_add") != "early_add":
        raise ValueError("FEATURE_SCALE is an early_add adapter, not separate matching.")


def feature_gain(cfg, branch):
    """Disabled/unit gains do not introduce a tensor operation or state key."""
    scale = cfg.MODEL.FEATURE_SCALE
    return float(scale[f"{branch.upper()}_GAIN"]) if scale.ENABLE else 1.0


def scale_feature(features, gain):
    # Do not modify the frozen feature cache or motion output in-place.
    return features if gain == 1.0 else features * gain


def validate_feature_scale_checkpoint(model, saved_cfg, allow_change=False):
    """Prevent silently evaluating/resuming weights at another feature scale.

    Legacy checkpoints without this field used unit gains. Explicit epoch reset
    permits initialization at another scale, but not silent optimizer resume.
    """
    import yaml

    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg.MODEL, "FEATURE_SCALE"):
        return
    if isinstance(saved_cfg, str):
        saved_cfg = yaml.safe_load(saved_cfg)
    saved = (saved_cfg or {}).get("MODEL", {}).get("FEATURE_SCALE", {})
    branches = ("appearance", "hod", "ctd")
    previous = tuple(float(saved.get(f"{b.upper()}_GAIN", 1.))
                     if saved.get("ENABLE", False) else 1. for b in branches)
    current = tuple(feature_gain(cfg, b) for b in branches)
    if previous != current and not allow_change:
        raise ValueError(
            f"FEATURE_SCALE checkpoint gains {previous} != configured gains {current}. "
            "Use the saved run config for evaluation/resume; changing scale for "
            "fresh initialization requires an explicit checkpoint epoch reset.")
