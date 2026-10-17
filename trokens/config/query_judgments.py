"""Resolve the independent quality and uniqueness switches."""

from typing import NamedTuple


class QueryJudgments(NamedTuple):
    uniqueness: bool
    quality: bool


def resolve_query_judgments(cfg):
    enabled = bool(getattr(cfg, "ENABLE", False))
    return QueryJudgments(
        uniqueness=enabled and bool(getattr(cfg, "UNIQUENESS_ENABLE", True)),
        quality=enabled and bool(getattr(cfg, "QUALITY_ENABLE", True)),
    )
