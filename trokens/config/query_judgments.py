"""Resolve Query judgment switches identically for routing and training loss."""

from typing import NamedTuple


class QueryJudgments(NamedTuple):
    uniqueness: bool
    quality: bool
    local_refinement: bool
    frame_verification: bool


def query_judgment_backend(cfg):
    """Legacy remains the default for checkpoints and other datasets."""
    backend = str(getattr(cfg, "BACKEND", "legacy")).lower()
    if backend not in {"legacy", "learned_dual_verifier"}:
        raise ValueError(f"Unknown QUERY_CLASS_MATCHABILITY.BACKEND: {backend!r}")
    return backend


def resolve_query_judgments(cfg):
    """Learned heads follow master flags; legacy keeps its subordinate options."""
    enabled = bool(getattr(cfg, "ENABLE", True))
    uniqueness = enabled and bool(getattr(cfg, "UNIQUENESS_ENABLE", True))
    learned = query_judgment_backend(cfg) == "learned_dual_verifier"
    if learned:
        if enabled and bool(getattr(cfg, "LOCAL_REFINEMENT_ENABLE", False)):
            raise ValueError("The learned verifier cannot rewrite the Query route via LOCAL_REFINEMENT.")
        return QueryJudgments(
            uniqueness=uniqueness,
            quality=enabled and bool(getattr(cfg, "QUALITY_ENABLE", True)),
            local_refinement=False,
            frame_verification=uniqueness,
        )
    return QueryJudgments(
        uniqueness=uniqueness,
        quality=(
            enabled
            and bool(getattr(cfg, "QUALITY_ENABLE", True))
            and bool(getattr(cfg, "ABSOLUTE_MASS_ENABLE", False))
        ),
        local_refinement=(
            uniqueness and bool(getattr(cfg, "LOCAL_REFINEMENT_ENABLE", False))
        ),
        frame_verification=(
            uniqueness and bool(getattr(cfg, "EVIDENCE_VERIFICATION_ENABLE", False))
        ),
    )
