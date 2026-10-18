"""Support text-only; Query Raw-positive-Support + text; matching stays Post."""

import copy

import pytest
import torch

from test_label_pre_temporal import _new_model, _episode
from trokens.models.label_conditioned_routing import (
    _initial_conditions, _prepare, _encode, label_conditioned_episode,
)
from trokens.utils.checkpoint import validate_label_pre_temporal_checkpoint


def _capture_conditions(model, features, raw, meta):
    calls = []
    hook = model.label_pre_temporal.register_forward_hook(
        lambda module, args, output: calls.append((args[1].detach().clone(), output.detach().clone())))
    try:
        output = label_conditioned_episode(model, features, raw, meta)
    finally:
        hook.remove()
    assert len(calls) == 1
    return output, *calls[0]


@pytest.mark.parametrize("interleaved", [False, True])
def test_episode_role_conditions_and_final_post_matching_oracle(interleaved):
    model = _new_model(classification=True).eval()
    features, raw, meta = _episode()
    if interleaved:
        order = torch.tensor([2, 0, 3, 1])
        features, raw = features[order], raw[order]
        meta = {k: v[order] if v.ndim and v.shape[0] == 4 else v for k, v in meta.items()}
    support = meta["support_mask"]
    si = support.nonzero().flatten()
    text = model._get_pot_label_text_features(meta["episode_class_ids"], raw.dtype)
    evidence = meta["pred_visibility"] & meta["pred_query_mask"]
    positive = meta["episode_positive_labels"][support]
    expected_query = _initial_conditions(model, raw, evidence, si, positive, text)
    result, conditions, gates = _capture_conditions(model, features, raw, meta)
    torch.testing.assert_close(conditions[support], text.expand(si.numel(), -1, -1))
    torch.testing.assert_close(conditions[~support], expected_query.expand((~support).sum(), -1, -1))
    assert not torch.allclose(expected_query, text)

    # Independent serial path: encode every trajectory with pure-text gates,
    # then text-read Post frames, then average known-positive Support videos.
    mask = meta["pred_query_mask"][support]
    tokens, token_mask, state = _prepare(model, features[support], mask)
    expected_gates = model.label_pre_temporal(raw[support], text, mask)
    torch.testing.assert_close(gates[support], expected_gates)
    prototypes = []
    for c in range(text.shape[0]):
        encoded = _encode(model, tokens, token_mask, state, expected_gates[:, c])
        assert encoded.shape == features[support].shape  # No trajectory compression.
        frames = model._compute_batched_frame_softmax_text_prototypes(
            encoded, mask, text[c:c+1])[0][:, 0]
        prototypes.append(frames[positive[:, c]].mean(0))
    expected = torch.stack(prototypes)
    torch.testing.assert_close(result["query_partial_support_prototypes"], expected)


def test_query_fusion_does_not_change_support_gates_or_matching(monkeypatch):
    model = _new_model(classification=True).eval()
    features, raw, meta = _episode()
    original, conditions, gates = _capture_conditions(model, features, raw, meta)
    fuse = model._fuse_episode_text_with_support_visual

    def altered_fusion(*args, **kwargs):
        condition, *rest = fuse(*args, **kwargs)
        return (-condition, *rest)

    monkeypatch.setattr(model, "_fuse_episode_text_with_support_visual", altered_fusion)
    changed, next_conditions, next_gates = _capture_conditions(model, features, raw, meta)
    support = meta["support_mask"]
    torch.testing.assert_close(conditions[support], next_conditions[support], atol=0, rtol=0)
    torch.testing.assert_close(gates[support], next_gates[support], atol=0, rtol=0)
    torch.testing.assert_close(original["query_partial_support_prototypes"],
                               changed["query_partial_support_prototypes"], atol=0, rtol=0)
    torch.testing.assert_close(original["global_classification_logits"][support],
                               changed["global_classification_logits"][support], atol=0, rtol=0)
    assert not torch.allclose(gates[~support], next_gates[~support])
    assert not torch.allclose(original["query_partial_query_prototypes"],
                              changed["query_partial_query_prototypes"])


def test_query_condition_uses_raw_support_not_encoded_features_or_raw_query():
    model = _new_model().eval()
    features, raw, meta = _episode()
    original, conditions, gates = _capture_conditions(model, features, raw, meta)
    changed_features = features.clone()
    changed_features[:2] = features[:2].roll(1, dims=-1) * 2
    changed, next_conditions, next_gates = _capture_conditions(model, changed_features, raw, meta)
    torch.testing.assert_close(conditions, next_conditions, atol=0, rtol=0)
    torch.testing.assert_close(gates, next_gates, atol=0, rtol=0)
    assert not torch.allclose(original["query_partial_support_prototypes"],
                              changed["query_partial_support_prototypes"])
    torch.testing.assert_close(original["query_partial_query_prototypes"],
                               changed["query_partial_query_prototypes"], atol=0, rtol=0)
    changed_raw = raw.clone()
    changed_raw[2:] = -raw[2:]
    _, next_conditions, _ = _capture_conditions(model, features, changed_raw, meta)
    torch.testing.assert_close(conditions, next_conditions, atol=0, rtol=0)
    changed_raw = raw.clone()
    changed_raw[:2] = -raw[:2]
    _, next_conditions, _ = _capture_conditions(model, features, changed_raw, meta)
    torch.testing.assert_close(conditions[:2], next_conditions[:2], atol=0, rtol=0)
    assert not torch.allclose(conditions[2:], next_conditions[2:])


def test_raw_condition_only_uses_known_positive_visible_support():
    model = _new_model().eval()
    features, raw, meta = _episode()
    meta["episode_positive_labels"] = meta["episode_positive_labels"].float()
    # Support 1 has an unknown label for class 0, not positive evidence.
    meta["episode_positive_labels"][1, 0] = -1
    meta["pred_visibility"][0, 0, 0] = False
    _, conditions, _ = _capture_conditions(model, features, raw, meta)
    changed = raw.clone()
    changed[1] *= -20
    changed[0, 0, 0] *= -20
    _, next_conditions, _ = _capture_conditions(model, features, changed, meta)
    torch.testing.assert_close(conditions[2:, 0], next_conditions[2:, 0], atol=0, rtol=0)
    # No known positive Support for class 0 => text-only fallback.
    meta["episode_label_known_mask"] = torch.ones(4, 2, dtype=torch.bool)
    meta["episode_label_known_mask"][0, 0] = False
    _, conditions, _ = _capture_conditions(model, features, changed, meta)
    torch.testing.assert_close(conditions[2:, 0], torch.tensor([[1., 0., 0., 0.]]).expand(2, -1))


def test_batched_role_gates_match_split_outputs_and_gradients_with_one_key_projection():
    a = _new_model().label_pre_temporal
    b = copy.deepcopy(a)
    _, raw, meta = _episode()
    raw_a, raw_b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
    conditions_a = torch.randn(4, 2, 4, requires_grad=True)
    conditions_b = conditions_a.detach().clone().requires_grad_()
    calls = []
    hook = a.key.register_forward_hook(lambda *args: calls.append(1))
    first = a(raw_a, conditions_a, meta["pred_query_mask"])
    hook.remove()
    second = torch.cat([b(raw_b[i:i+1], conditions_b[i], meta["pred_query_mask"][i:i+1])
                        for i in range(4)])
    assert len(calls) == 1
    torch.testing.assert_close(first, second)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(raw_a.grad, raw_b.grad)
    torch.testing.assert_close(conditions_a.grad, conditions_b.grad)
    assert conditions_a.grad[:2].abs().sum() > 0 and conditions_a.grad[2:].abs().sum() > 0
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("post_readout", [False, True])
@pytest.mark.parametrize("source_version", [1, 2, 3, 4, 5, 6, 7, 8, 99])
def test_routing_checkpoint_guards_changed_semantics(classification, post_readout, source_version):
    model = _new_model(classification=classification, post_readout=post_readout)
    destination = (7 if post_readout else 5) + int(classification)
    state = model.state_dict()
    state["label_pre_temporal.architecture_version"] = torch.tensor(source_version)
    if source_version == destination:
        assert not validate_label_pre_temporal_checkpoint(model, state)
    elif source_version == 99:
        with pytest.raises(ValueError, match="mismatch"):
            validate_label_pre_temporal_checkpoint(model, state, allow_init=True)
    else:
        with pytest.raises(ValueError, match="CHECKPOINT_EPOCH_RESET"):
            validate_label_pre_temporal_checkpoint(model, state)
        assert validate_label_pre_temporal_checkpoint(model, state, allow_init=True)
        assert "label_pre_temporal.architecture_version" not in state
        assert model.label_pre_temporal.architecture_version.item() == destination
