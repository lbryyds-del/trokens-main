"""CUDA AMP boundaries, FP32 fallback, masking, chunking and scaled gradients."""

import pytest
import torch

from test_learned_query_verifier import _direct_inputs, _model, _cfg
from tools.few_shot_multilabel import compute_learned_evidence_mil_loss
from trokens.models.query_evidence_verifier import FP32LayerNorm


def test_sav_fast_defaults_and_conservative_global_defaults():
    from trokens.config.defaults import get_cfg
    cfg = _cfg().FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    assert (cfg.QUERY_CHUNK_SIZE, cfg.ACTIVATION_CHECKPOINT, cfg.AMP_ENABLE) == (16, False, True)
    assert "QUALITY_MODE" not in cfg and "UNIQUENESS_MODE" not in cfg
    defaults = get_cfg().FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    assert (defaults.QUERY_CHUNK_SIZE, defaults.ACTIVATION_CHECKPOINT, defaults.AMP_ENABLE) == (4, True, False)
    assert "QUALITY_MODE" not in defaults and "UNIQUENESS_MODE" not in defaults


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("mask_case", ["partial", "empty", "no_confuser"])
def test_amp_outputs_masks_gradients_and_grad_scaler(checkpoint, detach, mask_case):
    torch.manual_seed(228)
    baseline = _model(checkpoint=True).cuda().query_evidence_verifier
    baseline.amp_enabled = False
    baseline.chunk_size = 4
    actual = _model(checkpoint=checkpoint).cuda().query_evidence_verifier
    actual.amp_enabled = True
    actual.chunk_size = 16
    baseline.detach_inputs = actual.detach_inputs = detach
    actual.load_state_dict(baseline.state_dict(), strict=True)
    assert list(baseline.state_dict()) == list(actual.state_dict())
    inputs = _direct_inputs("cuda")
    # Exercise a complete chunk AND a tail chunk, with partial observations.
    for key in ("raw_query", "post_query", "query_weights", "query_valid"):
        value = inputs[key]
        inputs[key] = value.detach().repeat(9, *([1] * (value.ndim - 1))).requires_grad_(value.requires_grad)
    inputs["query_valid"][0, 1] = False
    inputs["support_raw_valid"][0, 0] = False
    inputs["support_frame_valid"][0, 0, 0] = False
    if mask_case == "empty":
        inputs["query_valid"][:] = False
        inputs["support_raw_valid"][:] = False
        inputs["support_frame_valid"][:] = False
    elif mask_case == "no_confuser":
        inputs["support_labels"][:] = True
    copies = {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in inputs.items()}
    outputs = []
    labels = torch.eye(2, device="cuda").repeat(9, 1)
    for verifier, data in ((baseline, inputs), (actual, copies)):
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
        optimizer = torch.optim.SGD(verifier.parameters(), lr=0.01)
        with torch.autocast("cuda", dtype=torch.float16):
            result = verifier(**data)
            # Same head weights as SAV. No optimizer step or weight changes.
            loss = 0.05 * sum(compute_learned_evidence_mil_loss(
                result[name + "_logits"], result[name + "_valid"], labels,
            )["loss"] for name in ("quality", "relation"))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        outputs.append(result)
        for key, value in result.items():
            if value.is_floating_point():
                assert value.dtype == torch.float32, key
                assert torch.isfinite(value).all(), key
        for name, parameter in verifier.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        for key, value in data.items():
            if value.requires_grad:
                if detach:
                    assert value.grad is None, key
                else:
                    assert value.grad is not None and torch.isfinite(value.grad).all(), key
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=2e-3, rtol=5e-3, msg=key)
    for (name, left), (_, right) in zip(baseline.named_parameters(), actual.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-4, rtol=1e-2, msg=name)
    if not detach:
        for name, value in inputs.items():
            if value.requires_grad:
                torch.testing.assert_close(value.grad / 65536, copies[name].grad / 65536,
                                           atol=2e-4, rtol=1e-2, msg=name)
    if mask_case in ("empty", "no_confuser"):
        expected_quality = copies["query_valid"].any(-1)[:, None].expand_as(outputs[1]["quality"]).float()
        torch.testing.assert_close(outputs[1]["quality"], expected_quality, atol=0, rtol=0)
        assert torch.equal(outputs[1]["relation"], torch.ones_like(outputs[1]["relation"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp,outer_amp", [(True, True), (False, True), (True, False)])
def test_amp_linear_attention_and_fp32_normalization_boundaries(amp, outer_amp):
    verifier = _model().cuda().query_evidence_verifier
    verifier.amp_enabled = amp
    observed, hooks = {}, []
    for name, module in verifier.named_modules():
        if isinstance(module, (torch.nn.Linear, FP32LayerNorm)):
            hooks.append(module.register_forward_hook(
                lambda module, args, output, name=name: observed.__setitem__(name, output.dtype)))
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=outer_amp):
            output = verifier(**_direct_inputs("cuda"))
        for name, module in verifier.named_modules():
            if name in observed:
                expected = torch.float16 if amp and outer_amp and not isinstance(module, FP32LayerNorm) else torch.float32
                assert observed[name] == expected, (name, observed[name])
        assert all(value.dtype == torch.float32 for value in output.values() if value.is_floating_point())
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_invalid_nonfinite_inputs_are_sanitized_before_projection():
    verifier = _model().cuda().query_evidence_verifier
    inputs = _direct_inputs("cuda")
    with torch.no_grad():
        inputs["raw_query"][0, 0, 0, 0] = float("nan")
        inputs["raw_support"][0, 0, 0, 0] = float("inf")
        inputs["post_query"][0, 0, 0, 0] = -float("inf")
        inputs["post_references"][0, 0, 0, 0] = 1e10
    inputs["query_valid"][0, 0, 0] = False
    inputs["support_raw_valid"][0, 0, 0] = False
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = verifier(**inputs)
        loss = outputs["quality_logits"].square().mean() + outputs["relation_logits"].square().mean()
    loss.backward()
    assert all(torch.isfinite(v).all() for v in outputs.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in verifier.parameters())
