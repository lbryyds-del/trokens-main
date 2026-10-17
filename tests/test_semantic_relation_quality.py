"""Bounded semantic quality: exact anchors, conditional relations and isolation."""

import pytest
import torch
from torch.nn import functional as F

from test_learned_query_verifier import _cfg, _direct_inputs, _episode, _model, _run
from tools.few_shot_multilabel import compute_learned_evidence_mil_loss, get_query_matchability_metrics
from trokens.models.query_evidence_verifier import QueryEvidenceVerifier, _unit
from trokens.utils.checkpoint import load_checkpoint, validate_verifier_checkpoint


def _verifier():
    torch.manual_seed(129)
    return _model().query_evidence_verifier


def test_config_and_no_old_quality_classifier_or_class_specific_parameters():
    cfg = _cfg()
    vc = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    assert "QUALITY_MODE" not in vc and "UNIQUENESS_MODE" not in vc
    assert cfg.OUTPUT_DIR == "output/sav"
    assert not cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.CHECKPOINT_FILE_PATH == ""
    verifier = _verifier()
    assert verifier.architecture_version.item() == 3
    assert not hasattr(verifier, "raw_projection")
    assert not hasattr(verifier, "quality_head")
    assert not hasattr(verifier, "semantic_attention")
    head = verifier.semantic_quality
    for name in ("max_pool_delta", "max_reference_delta", "max_evidence_delta"):
        assert name in dict(head.named_buffers()) and name not in dict(head.named_parameters())
    assert sum(p.numel() for p in head.parameters()) > 4


def test_unprojected_raw_anchor_and_exact_final_formula():
    verifier, inputs = _verifier(), _direct_inputs()
    outputs = verifier(**inputs)
    cosine = torch.matmul(_unit(inputs["raw_query"]), _unit(inputs["text"]).T).permute(0, 3, 1, 2)
    expected = (outputs["quality_weights"] * cosine).sum(-1)
    torch.testing.assert_close(outputs["quality_anchor"], expected, atol=1e-6, rtol=1e-6)
    head = verifier.semantic_quality
    scale = F.softplus(head.scale_raw) + 1e-4
    torch.testing.assert_close(outputs["quality_logits"],
                               scale * (expected + outputs["quality_correction"]) + head.bias)
    assert (outputs["quality_correction"].abs() <= head.max_evidence_delta + 1e-7).all()
    assert (outputs["quality_pool_shift"] <= head.max_pool_delta + 1e-7).all()
    assert (expected >= cosine.min(-1).values - 1e-6).all()
    assert (expected <= cosine.max(-1).values + 1e-6).all()


def test_spatial_pool_is_semantic_softmax_plus_bounded_correction_and_masks():
    verifier, inputs = _verifier(), _direct_inputs()
    head = verifier.semantic_quality
    raw, text = _unit(inputs["raw_query"]), _unit(inputs["text"])
    valid = inputs["query_valid"].clone()
    valid[0, 1] = False
    valid[1, 0, 0] = False
    with torch.no_grad():
        head.pool[-1].weight.fill_(1000)
        head.pool[-1].bias.fill_(1000)
    weights, evidence, delta = head._pool(raw, head.projection(raw), text, head.projection(text), valid)
    assert (delta.abs() <= head.max_pool_delta).all()
    assert (weights[:, 0, 1] == 0).all() and (weights[:, 1, 0, 0] == 0).all()
    torch.testing.assert_close(weights.sum(-1), valid.any(-1)[None].expand(2, -1, -1).float())
    with torch.no_grad():
        head.pool[-1].weight.zero_()
        head.pool[-1].bias.zero_()
    weights, evidence, _ = head._pool(raw, head.projection(raw), text, head.projection(text), inputs["query_valid"])
    torch.testing.assert_close(weights, (evidence / head.patch_temperature).softmax(-1))


def test_support_references_are_original_raw_weighted_vectors_not_projected_values():
    head, inputs = _verifier().semantic_quality, _direct_inputs()
    raw, text = _unit(inputs["raw_support"]), _unit(inputs["text"])
    projected, projected_text = head.projection(raw), head.projection(text)
    valid = inputs["support_raw_valid"]
    weights, _, _ = head._pool(raw, projected, text, projected_text, valid)
    refs, projected_refs, text_cosine = head._references(raw, projected, text, projected_text, valid)
    expected = _unit((weights.unsqueeze(-1) * raw[None]).sum(-2)).flatten(1, 2)
    torch.testing.assert_close(refs, expected)
    torch.testing.assert_close(projected_refs, head.projection(expected))
    torch.testing.assert_close(text_cosine, (expected * text[:, None]).sum(-1))


def test_repeating_support_set_does_not_introduce_reference_count_prior():
    verifier, inputs = _verifier(), _direct_inputs()
    first = verifier(**inputs)
    repeated = dict(inputs)
    for key in ("raw_support", "post_references", "support_raw_valid", "support_frame_valid", "support_labels"):
        repeated[key] = torch.cat((inputs[key], inputs[key]), dim=0)
    second = verifier(**repeated)
    for key in ("quality_logits", "quality_anchor", "quality_correction", "relation_logits"):
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-6)


def test_neural_correction_cannot_exceed_bound_even_with_extreme_output_weights():
    verifier, inputs = _verifier(), _direct_inputs()
    head = verifier.semantic_quality
    for sign in (-1, 1):
        with torch.no_grad():
            head.relation[-1].weight.zero_()
            head.relation[-1].bias.fill_(sign * 10000.)
        output = verifier(**inputs)
        expected = torch.full_like(output["quality_correction"], sign * head.max_evidence_delta.item())
        torch.testing.assert_close(output["quality_correction"], expected)
        torch.testing.assert_close(output["quality_correction_saturation"], torch.ones_like(expected))


def test_support_relations_change_quality_at_identical_semantic_anchor_and_disable_cleanly():
    verifier, inputs = _verifier(), _direct_inputs()
    first = verifier(**inputs)
    swapped = verifier(**dict(inputs, support_labels=~inputs["support_labels"]))
    changed_inputs = dict(inputs, raw_support=-inputs["raw_support"])
    changed = verifier(**changed_inputs)
    for output in (swapped, changed):
        torch.testing.assert_close(first["quality_anchor"], output["quality_anchor"], atol=0, rtol=0)
        assert (first["quality_correction"] - output["quality_correction"]).abs().max() > 1e-6
    # This disables only the final relation correction, not private Raw pooling.
    verifier.semantic_quality.relation_strength = 0.
    baseline, no_support_effect = verifier(**inputs), verifier(**changed_inputs)
    assert baseline["quality_correction"].abs().sum() == 0
    torch.testing.assert_close(baseline["quality_logits"], no_support_effect["quality_logits"], atol=0, rtol=0)
    torch.testing.assert_close(first["relation_logits"], baseline["relation_logits"], atol=0, rtol=0)


def test_no_post_or_route_shortcut_and_invalid_support_values_do_not_affect_quality():
    verifier, inputs = _verifier(), _direct_inputs()
    inputs["support_raw_valid"][0, 1] = False
    inputs["query_valid"][0, 1, 0] = False
    first = verifier(**inputs)
    changed = {k: v.detach().clone() for k, v in inputs.items()}
    changed["raw_support"][0, 1] = 10000
    changed["raw_query"][0, 1, 0] = -10000
    changed["post_query"].neg_()
    changed["post_references"].neg_()
    changed["query_weights"] = torch.rand_like(changed["query_weights"]).softmax(-1)
    second = verifier(**changed)
    torch.testing.assert_close(first["quality_logits"], second["quality_logits"], atol=1e-6, rtol=1e-6)
    # Text matters even though class IDs are never passed to this head.
    changed["text"].neg_()
    assert not torch.allclose(first["quality_logits"], verifier(**changed)["quality_logits"])


def test_quality_ablation_preserves_rho_and_all_base_route_outputs():
    torch.manual_seed(321)
    enabled = _model(quality=True).eval()
    torch.manual_seed(321)
    disabled = _model(quality=False).eval()
    post, raw, meta = _episode()
    first, second = _run(enabled, post, raw, meta), _run(disabled, post, raw, meta)
    for key in ("query_relation_logits", "query_partial_query_prototypes",
                "query_partial_support_prototypes", "query_partial_q2s_base_logits",
                "query_patch_conditional_weights"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0, msg=key)
    assert "query_quality_logits" not in second


def test_every_quality_parameter_learns_from_video_mil_not_hand_score_targets():
    verifier, inputs = _verifier(), _direct_inputs()
    head = verifier.semantic_quality
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0)
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    result = verifier(**inputs)
    loss = compute_learned_evidence_mil_loss(result["quality_logits"], result["quality_valid"], torch.eye(2))["loss"]
    loss.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in head.named_parameters():
        assert not torch.equal(before[name], parameter), name


def test_new_diagnostics_are_exported_with_positive_negative_split_and_detached():
    model = _model()
    post, raw, meta = _episode()
    output = _run(model, post, raw, meta)
    metrics = get_query_matchability_metrics(output, meta["episode_positive_labels"][2:].float())
    for name in ("anchor", "correction", "correction_abs", "correction_saturation", "pool_shift"):
        for sign in ("positive", "negative"):
            value = metrics[f"verifier_quality_{name}_{sign}"]
            assert torch.isfinite(value) and not value.requires_grad
    assert metrics["verifier_quality_scale"].item() == pytest.approx(10.)
    assert metrics["verifier_quality_max_evidence_delta"].item() == pytest.approx(.10)


@pytest.mark.parametrize("old_version", [1, 2, 4])
@pytest.mark.parametrize("allow_init", [False, True])
def test_incompatible_evidence_checkpoint_is_rejected(tmp_path, old_version, allow_init):
    target = _model()
    state = {k: v.clone() for k, v in target.state_dict().items()}
    state["query_evidence_verifier.architecture_version"] = torch.tensor(old_version)
    with pytest.raises(ValueError, match="architecture"):
        validate_verifier_checkpoint(target, state, allow_init=allow_init)
    path = tmp_path / "checkpoint.pyth"
    torch.save(dict(model_state=state, epoch=19), path)
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(str(path), target, data_parallel=False, allow_verifier_init=allow_init)


@pytest.mark.parametrize("field,value", [
    ("QUALITY_HIDDEN_DIM", 0), ("QUALITY_PATCH_TEMPERATURE", 0.),
    ("QUALITY_REFERENCE_TEMPERATURE", float("nan")), ("QUALITY_MAX_POOL_DELTA", -1.),
    ("QUALITY_MAX_REFERENCE_DELTA", float("inf")), ("QUALITY_MAX_EVIDENCE_DELTA", -.1),
    ("QUALITY_SCALE_INIT", 0.), ("QUALITY_BIAS_INIT", float("nan")), ("QUALITY_RELATION_STRENGTH", 2.),
])
def test_invalid_quality_config_fails_early(field, value):
    cfg = _cfg().FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        QueryEvidenceVerifier(4, cfg)
