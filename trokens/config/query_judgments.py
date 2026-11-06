"""Resolve the rho verification switch."""

from typing import NamedTuple


class QueryJudgments(NamedTuple):
    uniqueness: bool


def resolve_query_judgments(cfg):
    enabled = bool(getattr(cfg, "ENABLE", False))
    return QueryJudgments(
        uniqueness=enabled and bool(getattr(cfg, "UNIQUENESS_ENABLE", True)),
    )
