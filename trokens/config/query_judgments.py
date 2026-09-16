"""Resolve Query judgment switches identically for routing and training loss."""

from typing import NamedTuple


class QueryJudgments(NamedTuple):
    uniqueness: bool
    quality: bool
    local_refinement: bool
    frame_verification: bool


def resolve_query_judgments(cfg):
    """Master flags gate legacy options without changing older configs."""
    enabled = bool(getattr(cfg, "ENABLE", True))
    uniqueness = enabled and bool(getattr(cfg, "UNIQUENESS_ENABLE", True))
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
