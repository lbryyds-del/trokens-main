"""Pure implementation optimization against the frozen sequential verifier."""

from collections import Counter

import pytest
import torch

from reference_query_verifier import SequentialQueryEvidenceVerifier
from test_learned_query_verifier import _cfg, _direct_inputs, _episode, _model, _run
from trokens.models.query_evidence_verifier import QueryEvidenceVerifier
from tools.few_shot_multilabel import compute_query_partial_q2s_loss


def _clone_inputs(inputs):
    return {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in inputs.items()}


@pytest.mark.parametrize("quality,uniqueness", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("mask_case", ["partial", "empty", "no_confuser"])
def test_forward_and_all_gradients_match_sequential(quality, uniqueness, checkpoint, detach, mask_case):
    cfg = _cfg().FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    cfg.UNIQUENESS_MODE = "relation_attention"
    cfg.QUALITY_MODE = "semantic_attention"
    cfg.HIDDEN_DIM, cfg.NUM_HEADS, cfg.FFN_DIM = 8, 2, 16
    cfg.QUERY_CHUNK_SIZE = 1
    cfg.ACTIVATION_CHECKPOINT, cfg.DETACH_INPUTS = checkpoint, detach
    reference = SequentialQueryEvidenceVerifier(4, cfg, quality, uniqueness)
    optimized = QueryEvidenceVerifier(4, cfg, quality, uniqueness)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(optimized.state_dict())
    inputs = _direct_inputs()
    inputs["query_valid"][0, 1] = False
    inputs["support_raw_valid"][0, 0, 1:] = False
    inputs["support_frame_valid"][0, 0, 0] = False
    inputs["support_known"] = torch.tensor([[True, True], [True, False]])
    if mask_case == "empty":
        inputs["query_valid"][:] = False
        inputs["support_raw_valid"][:] = False
        inputs["support_frame_valid"][:] = False
    elif mask_case == "no_confuser":
        inputs["support_labels"][:] = True
    other_inputs = _clone_inputs(inputs)
    outputs = reference(**inputs), optimized(**other_inputs)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=2e-6, rtol=2e-5, msg=key)
    for result in outputs:
        # Exercise both supervised heads and nonlinear gate paths.
        loss = sum(value.square().mean() for key, value in result.items()
                   if key.endswith("logits") or key in ("quality", "relation"))
        loss.backward()
    for (name, left), (other_name, right) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == other_name
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-5, msg=name)
    for name, value in inputs.items():
        actual = other_inputs[name]
        if value.requires_grad:
            if value.grad is None:
                assert actual.grad is None, name
            else:
                torch.testing.assert_close(value.grad, actual.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_reuses_qkv_projections_without_cross_forward_cache():
    verifier = _model(checkpoint=True, mode="relation_attention").query_evidence_verifier
    counts, hooks = Counter(), []
    for name, module in verifier.named_modules():
        if name.endswith(("attention.q", "attention.k", "attention.v")):
            hooks.append(module.register_forward_hook(
                lambda module, args, output, name=name: counts.update([name])))
    inputs = _direct_inputs()
    try:
        for _ in range(2):
            counts.clear()
            verifier(**inputs)
            assert counts == {
                "semantic_attention.q": 2,  # two Query chunks, all classes/signs batched
                "semantic_attention.k": 1, "semantic_attention.v": 1,
                "relation_attention.q": 1,
                "relation_attention.k": 1, "relation_attention.v": 1,
            }
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_matched_route_gates_losses_and_amp_gradients(device):
    reference = _model(checkpoint=True, mode="relation_attention").to(device)
    optimized = _model(checkpoint=True, mode="relation_attention").to(device)
    # This test isolates the equivalent batched rewrite, not FP16 rounding.
    optimized.query_evidence_verifier.amp_enabled = False
    cfg = reference.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    reference.query_evidence_verifier = SequentialQueryEvidenceVerifier(4, cfg).to(device)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    post, raw, metadata = _episode(device)
    inputs = [post.detach().clone().requires_grad_() for _ in range(2)]
    results, losses = [], []
    for model, value in zip((reference, optimized), inputs):
        with torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
            output = _run(model, value, raw, metadata)
            loss, _, _ = compute_query_partial_q2s_loss(
                output["query_partial_q2s_logits"], metadata["episode_positive_labels"][2:].float(), output, model.cfg)
        results.append(output)
        losses.append(loss)
        loss.backward()
    for key in ("query_quality_logits", "query_relation_logits", "query_frame_target_mass",
                "query_frame_confuser_mass", "query_frame_null_mass", "query_partial_q2s_logits"):
        torch.testing.assert_close(results[0][key], results[1][key], atol=3e-6, rtol=3e-5, msg=key)
    torch.testing.assert_close(losses[0], losses[1], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=3e-6, rtol=3e-5)
    for (name, left), (_, right) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-5, msg=name)
