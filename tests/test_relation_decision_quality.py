"""Ported 9.1 quality-only experiment: learned decisions, unchanged readers/matcher."""

import pytest
import torch

from test_learned_query_verifier import _cfg, _direct_inputs, _episode, _model, _run
from tools.few_shot_multilabel import (
    compute_learned_evidence_mil_loss, compute_query_partial_q2s_loss,
    get_query_matchability_metrics,
)
from trokens.models.query_evidence_verifier import QueryEvidenceVerifier, _unit
from trokens.utils.checkpoint import validate_verifier_checkpoint


def _new(**kwargs):
    return _model(quality_mode="relation_decision", **kwargs)


def test_sav_selects_quality_only_experiment_and_preserves_training_contract():
    cfg = _cfg()
    matchability = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    verifier = matchability.LEARNED_VERIFIER
    assert cfg.OUTPUT_DIR == "output/sav_relation_decision"
    assert verifier.QUALITY_MODE == "relation_decision"
    assert "UNIQUENESS_MODE" not in verifier  # This branch only retains metric rho.
    assert (verifier.QUALITY_HIDDEN_DIM, verifier.QUALITY_DECISION_LOCAL_DIM,
            verifier.QUALITY_DECISION_READOUT_DIM) == (64, 32, 16)
    assert (matchability.BASE_LOGIT_LOSS_WEIGHT, matchability.VERIFIED_LOGIT_LOSS_WEIGHT,
            verifier.SEMANTIC_MIL_WEIGHT, verifier.RELATION_MIL_WEIGHT) == (.60, .40, .05, .05)
    assert (verifier.QUERY_CHUNK_SIZE, verifier.ACTIVATION_CHECKPOINT, verifier.AMP_ENABLE) == (16, False, True)
    assert not cfg.TRAIN.AUTO_RESUME and not cfg.TRAIN.CHECKPOINT_FILE_PATH
    assert cfg.TRAIN.NEW_TRAIN
    assert not cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE  # Selected by derived early-routing configs.
    assert verifier.DETACH_INPUTS and matchability.ABSOLUTE_MASS_ONE_SIDED


def test_head_has_no_mandatory_anchor_or_unused_residual_parameters():
    verifier = _new().query_evidence_verifier
    assert verifier.architecture_version.item() == 5
    head = verifier.semantic_quality
    for name in ("scale_raw", "bias", "max_evidence_delta", "relation_strength", "relation"):
        assert not hasattr(head, name), name
    inputs = _direct_inputs()
    # A constant network decision remains constant despite different raw
    # cosines. This would NOT hold for v3's mandatory anchor + correction.
    with torch.no_grad():
        head.decision[-1].weight.zero_()
        head.decision[-1].bias.fill_(-2.)
    first = verifier(**inputs)
    changed = verifier(**dict(inputs, raw_query=-inputs["raw_query"]))
    for result in (first, changed):
        torch.testing.assert_close(result["quality_logits"], torch.full_like(result["quality_logits"], -2.))
        assert "quality_anchor" not in result and "quality_correction" not in result
    assert not torch.allclose(first["quality_semantic_evidence"], changed["quality_semantic_evidence"])


def test_shared_reader_matches_v3_exactly_with_identical_reader_weights():
    old = _model(quality_mode="semantic_relation").query_evidence_verifier.semantic_quality
    new = _new().query_evidence_verifier.semantic_quality
    new.projection.load_state_dict(old.projection.state_dict())
    new.pool.load_state_dict(old.pool.state_dict())
    data = _direct_inputs()
    raw, support, text = (_unit(data[k]) for k in ("raw_query", "raw_support", "text"))
    valid, sv = data["query_valid"], data["support_raw_valid"]
    mask = torch.stack([(sv.any(-1)[None] & sign.T[:, :, None]).flatten(1)
                        for sign in (data["support_labels"], ~data["support_labels"])], 1)
    results = []
    for head in (old, new):
        references = head._references(support, head.projection(support), text, head.projection(text), sv)
        results.append(head._relation_inputs(raw, head.projection(raw), text, head.projection(text),
                                            valid, *references, mask))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("uniqueness", [False, True])
def test_switches_keep_routing_rho_and_base_logits_unchanged(quality, uniqueness):
    old = _model(quality=quality, uniqueness=uniqueness, quality_mode="semantic_relation")
    new = _new(quality=quality, uniqueness=uniqueness)
    post, raw, meta = _episode()
    post.requires_grad_()
    before, after = _run(old, post, raw, meta), _run(new, post, raw, meta)
    for key in ("query_partial_query_prototypes", "query_partial_support_prototypes",
                "query_partial_q2s_base_logits", "query_patch_conditional_weights", "query_relation_logits"):
        if key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    labels = meta["episode_positive_labels"][2:].float()
    loss, _, metrics = compute_query_partial_q2s_loss(after["query_partial_q2s_logits"], labels, after, new.cfg)
    assert ("q2s_quality_mil_loss" in metrics) == quality
    loss.backward()
    assert torch.isfinite(post.grad).all()
    if new.query_evidence_verifier is not None:
        for name, parameter in new.query_evidence_verifier.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    if quality:
        torch.testing.assert_close(after["query_frame_target_mass"],
                                   after["query_frame_absolute_patch_mass"] * after.get("query_frame_matchability", 1.))
        assert (after["query_partial_q2s_logits"] <= after["query_partial_q2s_base_logits"] + 1e-6).all()
        reported = get_query_matchability_metrics(after, labels)
        assert "verifier_quality_semantic_evidence_positive" in reported
        assert "verifier_quality_relation_feature_norm_negative" in reported
        assert not any("quality_correction" in key or "quality_scale" in key for key in reported)


def test_all_quality_parameters_train_but_detached_inputs_do_not():
    torch.manual_seed(53)
    verifier, inputs = _new().query_evidence_verifier, _direct_inputs()
    head = verifier.semantic_quality
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0)
    before = {name: p.detach().clone() for name, p in head.named_parameters()}
    result = verifier(**inputs)
    loss = compute_learned_evidence_mil_loss(result["quality_logits"], result["quality_valid"], torch.eye(2))["loss"]
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    for value in inputs.values():
        if value.requires_grad:
            assert value.grad is None
    optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in head.named_parameters())


@pytest.mark.parametrize("case", ["empty_query", "empty_support", "no_negative", "unknown"])
def test_missing_evidence_is_finite_and_not_false_confidence(case):
    verifier, inputs = _new().query_evidence_verifier, _direct_inputs()
    if case == "empty_query":
        inputs["query_valid"][:] = False
    elif case == "empty_support":
        inputs["support_raw_valid"][:] = False
    elif case == "no_negative":
        inputs["support_labels"][:] = True
    else:
        inputs["support_labels"] = torch.full_like(inputs["support_labels"], -1, dtype=torch.float32)
    out = verifier(**inputs)
    assert all(torch.isfinite(value).all() for value in out.values())
    assert not out["quality_valid"].any()
    assert (out["quality"] == (0 if case == "empty_query" else 1)).all()
    loss = compute_learned_evidence_mil_loss(out["quality_logits"], out["quality_valid"], torch.eye(2))["loss"]
    assert loss == 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in verifier.semantic_quality.parameters())


def test_query_truth_is_not_a_forward_input_and_support_order_is_invariant():
    model = _new().eval()
    post, raw, meta = _episode()
    first = _run(model, post, raw, meta)
    changed = dict(meta, episode_positive_labels=meta["episode_positive_labels"].clone())
    changed["episode_positive_labels"][2:] = ~changed["episode_positive_labels"][2:]
    second = _run(model, post, raw, changed)
    order = torch.tensor([1, 0, 2, 3])
    permuted = {k: (v[order] if v.ndim and v.shape[0] == 4 else v) for k, v in meta.items()}
    third = _run(model, post[order], raw[order], permuted)
    for key in ("query_quality_logits", "query_relation_logits", "query_partial_q2s_logits"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        torch.testing.assert_close(first[key], third[key], atol=2e-6, rtol=2e-6)


def test_quality_reads_support_vectors_not_post_or_only_query_scores():
    torch.manual_seed(54)
    verifier, inputs = _new().query_evidence_verifier, _direct_inputs()
    first = verifier(**inputs)
    changed = verifier(**dict(inputs, raw_support=-inputs["raw_support"]))
    torch.testing.assert_close(first["quality_semantic_evidence"], changed["quality_semantic_evidence"], atol=0, rtol=0)
    assert not torch.equal(first["quality_logits"], changed["quality_logits"])
    ignored = verifier(**dict(inputs, post_query=-inputs["post_query"],
                             post_references=-inputs["post_references"],
                             query_weights=torch.rand_like(inputs["query_weights"]).softmax(-1)))
    torch.testing.assert_close(first["quality_logits"], ignored["quality_logits"], atol=0, rtol=0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_chunk_and_checkpoint_equivalence(checkpoint):
    verifier, inputs = _new(checkpoint=checkpoint).query_evidence_verifier, _direct_inputs()
    first = verifier(**inputs)
    verifier.chunk_size = 16
    second = verifier(**inputs)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-6)
    second["quality_logits"].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in verifier.semantic_quality.parameters())


@pytest.mark.parametrize("allow_init", [False, True])
def test_v3_and_new_quality_checkpoints_cannot_be_mixed(allow_init):
    old, new = _model(quality_mode="semantic_relation"), _new()
    for target, source in ((new, old), (old, new)):
        with pytest.raises(ValueError, match="version"):
            validate_verifier_checkpoint(target, source.state_dict(), allow_init=allow_init)


@pytest.mark.parametrize("field,value", [
    ("QUALITY_DECISION_LOCAL_DIM", 0), ("QUALITY_DECISION_READOUT_DIM", -1),
    ("QUALITY_HIDDEN_DIM", 0), ("QUALITY_PATCH_TEMPERATURE", 0.),
    ("QUALITY_REFERENCE_TEMPERATURE", float("nan")), ("QUALITY_MAX_POOL_DELTA", -1.),
])
def test_bad_configuration_fails_before_training(field, value):
    cfg = _cfg().FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    cfg.QUALITY_MODE = "relation_decision"
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        QueryEvidenceVerifier(4, cfg)
