"""Metric-anchored rho: evidence semantics, isolation and version safety."""

import pytest
import torch
from torch.nn import functional as F

from test_learned_query_verifier import _direct_inputs, _episode, _model, _run
from tools.few_shot_multilabel import compute_learned_evidence_mil_loss, get_query_matchability_metrics
from trokens.models.query_evidence_verifier import QueryEvidenceVerifier
from trokens.utils.checkpoint import load_checkpoint, validate_verifier_checkpoint


def test_metric_defaults_and_only_two_shared_calibration_parameters():
    model = _model(quality=False)
    verifier = model.query_evidence_verifier
    assert verifier.architecture_version.item() == 2
    assert set(dict(verifier.named_parameters())) == {"metric_temperature_raw", "metric_scale_raw"}
    assert F.softplus(verifier.metric_temperature_raw).item() + 1e-4 == pytest.approx(0.1)
    assert F.softplus(verifier.metric_scale_raw).item() + 1e-4 == pytest.approx(10.0)


def test_aggregate_monotonicity_count_normalization_mask_and_learned_temperature():
    verifier = _model(quality=False).query_evidence_verifier
    scores = torch.tensor([[[[[0.15, 0.65, -0.5]]]]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    result = verifier._metric_aggregate(scores, valid)
    expected = 0.1 * (torch.logsumexp(scores[..., :2] / 0.1, -1) - torch.tensor(2.).log())
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(verifier._metric_aggregate(scores + 0.2, valid), result + 0.2)
    # Repeating the entire set does not introduce a reference-count prior.
    torch.testing.assert_close(verifier._metric_aggregate(scores.repeat(1, 1, 1, 1, 3), valid.repeat(1, 3)), result)
    changed = scores.detach().clone()
    changed[..., 2] = 10000
    torch.testing.assert_close(verifier._metric_aggregate(changed, valid), result)
    result.sum().backward()
    assert (scores.grad[..., :2] > 0).all()
    assert scores.grad[..., 2].item() == 0
    assert verifier.metric_temperature_raw.grad.abs().item() > 0
    # Same evidence at every reference must give that evidence, not tau*log R.
    for count in (1, 3, 17):
        equal = torch.full((1, 1, 1, 1, count), 0.3)
        torch.testing.assert_close(verifier._metric_aggregate(equal, torch.ones(1, count).bool()),
                                   torch.full((1, 1, 1, 1), 0.3))


def test_single_reference_matches_cosine_margin_and_has_no_text_shortcut():
    verifier = _model(quality=False).query_evidence_verifier
    inputs = _direct_inputs()
    inputs["support_frame_valid"][:, :, 1:] = False
    output = verifier(**inputs)
    query = F.normalize(inputs["post_query"], dim=-1)
    refs = F.normalize(inputs["post_references"], dim=-1)
    expected = []
    for k in range(2):
        pos = (query * refs[k, k, 0]).sum(-1)
        neg = (query * refs[1 - k, k, 0]).sum(-1)
        expected.append(10 * (inputs["query_weights"][:, k] * (pos - neg)).sum(-1))
    torch.testing.assert_close(output["relation_logits"], torch.stack(expected, 1), atol=2e-6, rtol=2e-6)
    changed = dict(inputs, raw_query=inputs["raw_query"] * 100,
                   text=torch.randn_like(inputs["text"]) * 10)
    torch.testing.assert_close(verifier(**changed)["relation_logits"], output["relation_logits"], atol=0, rtol=0)
    # Altering a different class cannot change this class's metric evidence.
    refs = inputs["post_references"].detach().clone()
    refs[:, 1] *= -1
    changed = verifier(**dict(inputs, post_references=refs))
    torch.testing.assert_close(changed["relation_logits"][:, 0], output["relation_logits"][:, 0], atol=0, rtol=0)


def test_metric_preserves_quality_parameters_outputs_and_all_base_route_tensors():
    torch.manual_seed(123)
    old = _model(mode="relation_attention").eval()
    torch.manual_seed(123)
    new = _model().eval()
    # Quality is initialized before either rho variant and has identical RNG.
    old_state = old.state_dict()
    for key, value in new.state_dict().items():
        if key in old_state and not key.endswith("architecture_version"):
            torch.testing.assert_close(value, old_state[key], atol=0, rtol=0)
    data = _episode()
    before, after = _run(old, *data), _run(new, *data)
    for key in ("query_quality_logits", "query_quality_patch_weights", "query_frame_absolute_patch_mass",
                "query_partial_query_prototypes", "query_partial_support_prototypes",
                "query_partial_q2s_base_logits", "query_patch_conditional_weights", "query_evidence_patch_weights"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0, msg=key)
    assert not torch.allclose(before["query_relation_logits"], after["query_relation_logits"])


def test_metric_temperature_and_scale_really_update_from_mil():
    verifier = _model(quality=False).query_evidence_verifier
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=0)
    initial = {k: v.detach().clone() for k, v in verifier.named_parameters()}
    out = verifier(**_direct_inputs())
    loss = compute_learned_evidence_mil_loss(out["relation_logits"], out["relation_valid"], torch.eye(2))["loss"]
    loss.backward()
    for parameter in verifier.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().item() > 0
    optimizer.step()
    for name, parameter in verifier.named_parameters():
        assert not torch.equal(parameter, initial[name])


def test_calibration_diagnostics_are_logged_without_gradients():
    model = _model()
    post, raw, meta = _episode()
    result = _run(model, post, raw, meta)
    metrics = get_query_matchability_metrics(result, meta["episode_positive_labels"][2:].float())
    for name, expected in (("metric_temperature", 0.1), ("metric_scale", 10.0)):
        value = metrics["verifier_" + name]
        assert value.item() == pytest.approx(expected)
        assert not value.requires_grad
    old = _run(_model(mode="relation_attention"), post, raw, meta)
    assert "query_metric_temperature" not in old


@pytest.mark.parametrize("source,target", [("relation_attention", "metric"), ("metric", "relation_attention")])
@pytest.mark.parametrize("allow_init", [False, True])
def test_cross_architecture_checkpoints_are_rejected_even_for_explicit_init(tmp_path, source, target, allow_init):
    original, current = _model(mode=source), _model(mode=target)
    state = original.state_dict()
    before = {k: v.clone() for k, v in current.state_dict().items()}
    with pytest.raises(ValueError, match="UNIQUENESS_MODE"):
        validate_verifier_checkpoint(current, state, allow_init=allow_init)
    path = tmp_path / "old.pyth"
    torch.save({"model_state": state, "epoch": 19, "optimizer_state": {}}, path)
    for inflate in (False, True):
        with pytest.raises(ValueError, match="UNIQUENESS_MODE"):
            load_checkpoint(str(path), current, data_parallel=False,
                            inflation=inflate, allow_verifier_init=allow_init)
    for key, value in current.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


@pytest.mark.parametrize("field,value", [("UNIQUENESS_MODE", "typo"), ("METRIC_TEMPERATURE_INIT", 0.),
                                       ("METRIC_SCALE_INIT", -1.), ("METRIC_SCALE_INIT", float("nan"))])
def test_bad_metric_config_fails_early(field, value):
    model = _model()
    cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.clone()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        QueryEvidenceVerifier(4, cfg)
