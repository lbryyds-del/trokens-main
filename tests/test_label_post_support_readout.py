"""Raw-conditioned encoding / Post-Support-conditioned final Query readout."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_label_asymmetric_conditions import _capture_conditions
from test_label_pre_temporal import _new_model, _episode, _loss, _single_path_ddp_worker
from trokens.config.defaults import get_cfg
import trokens.models.label_conditioned_routing as routing
from trokens.utils.checkpoint import load_checkpoint, validate_label_pre_temporal_checkpoint


def test_post_readout_config_changes_only_readout_and_output_directory():
    root = Path(__file__).resolve().parents[1] / "configs/trokens"
    control, experiment = get_cfg(), get_cfg()
    control.merge_from_file(str(root / "sav_label_pre_temporal_fast.yaml"))
    experiment.merge_from_file(str(root / "sav_label_pre_temporal_post_readout.yaml"))
    assert not control.FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT
    assert experiment.FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT
    assert control.OUTPUT_DIR != experiment.OUTPUT_DIR
    experiment.FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT = False
    experiment.OUTPUT_DIR = control.OUTPUT_DIR
    assert experiment == control


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("class_chunk", [1, 2])
def test_post_readout_matches_independent_prototype_fusion_oracle(interleaved, class_chunk):
    model = _new_model(classification=True, post_readout=True).eval()
    model.label_pre_temporal.class_chunk_size = class_chunk
    features, raw, meta = _episode()
    if interleaved:
        order = torch.tensor([2, 0, 3, 1])
        features, raw = features[order], raw[order]
        meta = {k: v[order] if v.ndim and v.shape[0] == 4 else v for k, v in meta.items()}
    output, _, gates = _capture_conditions(model, features, raw, meta)
    text = model._get_pot_label_text_features(meta["episode_class_ids"], raw.dtype)
    support = meta["support_mask"]
    mask = meta["pred_query_mask"]
    # Independent Support encoding -> pure-text frame readout -> positive mean.
    tokens, token_mask, state = routing._prepare(model, features[support], mask[support])
    prototypes = []
    for c in range(len(text)):
        post = routing._encode(model, tokens, token_mask, state, gates[support, c])
        frames = model._compute_batched_frame_softmax_text_prototypes(post, mask[support], text[c:c+1])[0]
        positive = meta["episode_positive_labels"][support, c]
        prototypes.append(frames[positive, 0].mean(0))
    prototypes = torch.stack(prototypes)
    torch.testing.assert_close(output["query_partial_support_prototypes"], prototypes)
    valid = prototypes.norm(dim=-1) > 1e-12
    visual = (prototypes * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
    condition = F.normalize(F.normalize(text, dim=-1) + F.normalize(visual, dim=-1), dim=-1)
    # Same Query encoding, but independently compute Post-conditioned readout.
    tokens, token_mask, state = routing._prepare(model, features[~support], mask[~support])
    expected = []
    for c in range(len(text)):
        post = routing._encode(model, tokens, token_mask, state, gates[~support, c])
        frames = model._compute_batched_frame_softmax_text_prototypes(
            post, mask[~support], condition[c:c+1])[0]
        expected.append(frames)
    expected = torch.cat(expected, 1)
    torch.testing.assert_close(output["query_partial_query_prototypes"], expected)
    logits = (model.pot_route_cfg.QUERY_PARTIAL_LOGIT_ALPHA
              * model._compute_bidirectional_frame_similarity(expected, prototypes)
              + model.pot_route_cfg.QUERY_PARTIAL_LOGIT_BIAS)
    torch.testing.assert_close(output["query_partial_q2s_base_logits"], logits)


def test_readout_switch_preserves_encoding_references_quality_and_global_logits(monkeypatch):
    model = _new_model(classification=True).eval()
    features, raw, meta = _episode()
    encoded, references = [], []
    encode = routing._encode_with_global

    def record_encode(*args):
        result = encode(*args)
        encoded.append(result[0].detach().clone())
        return result

    def record_references(module, args, kwargs):
        references.append(kwargs["post_references"].detach().clone())

    monkeypatch.setattr(routing, "_encode_with_global", record_encode)
    hook = model.query_evidence_verifier.register_forward_pre_hook(record_references, with_kwargs=True)
    try:
        before, condition, gate = _capture_conditions(model, features, raw, meta)
        count = len(encoded)
        reference_count = len(references)
        model.label_pre_temporal.post_support_readout = True
        after, next_condition, next_gate = _capture_conditions(model, features, raw, meta)
    finally:
        hook.remove()
    torch.testing.assert_close(condition, next_condition, atol=0, rtol=0)
    torch.testing.assert_close(gate, next_gate, atol=0, rtol=0)
    assert len(encoded) == 2 * count and len(references) == 2 * reference_count
    for a, b in zip(encoded[:count], encoded[count:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(references[:reference_count], references[reference_count:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in ("query_partial_support_prototypes", "global_classification_logits",
                "query_quality_logits", "query_quality_valid", "query_frame_absolute_patch_mass"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0, msg=key)
    for key in ("query_partial_query_prototypes", "query_partial_q2s_base_logits"):
        assert not torch.allclose(before[key], after[key]), key
    # The control switch is a real ablation, not an alias to the new path.
    model.label_pre_temporal.post_support_readout = False
    restored = routing.label_conditioned_episode(model, features, raw, meta)
    for key, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, restored[key], atol=0, rtol=0, msg=key)


def test_post_support_changes_readout_not_raw_encoding_condition():
    model = _new_model(classification=True, post_readout=True).eval()
    features, raw, meta = _episode()
    before, condition, gate = _capture_conditions(model, features, raw, meta)
    changed = features.clone()
    changed[:2] = features[:2].roll(1, dims=-1) * 2
    after, next_condition, next_gate = _capture_conditions(model, changed, raw, meta)
    torch.testing.assert_close(condition, next_condition, atol=0, rtol=0)
    torch.testing.assert_close(gate, next_gate, atol=0, rtol=0)
    torch.testing.assert_close(before["global_classification_logits"][2:],
                               after["global_classification_logits"][2:], atol=0, rtol=0)
    for key in ("query_partial_support_prototypes", "query_partial_query_prototypes"):
        assert not torch.allclose(before[key], after[key]), key


def test_post_condition_isolated_from_query_inputs_truth_and_nonpositive_support(monkeypatch):
    model = _new_model(post_readout=True).eval()
    model.label_pre_temporal.class_chunk_size = 2
    features, raw, meta = _episode()
    meta["episode_positive_labels"] = meta["episode_positive_labels"].float()
    # Explicitly unknown class 0 on Support 1 is not positive evidence.
    meta["episode_positive_labels"][1, 0] = -1
    conditions = []
    fusion = model._fuse_episode_text_with_support_visual

    def record(*args):
        result = fusion(*args)
        conditions.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(model, "_fuse_episode_text_with_support_visual", record)
    first = routing.label_conditioned_episode(model, features, raw, meta)
    assert len(conditions) == 2  # Raw first; then Post for this class group.
    changed_meta = copy.deepcopy(meta)
    changed_meta["episode_positive_labels"][2:] = float("nan")
    changed_meta["episode_label_known_mask"] = torch.ones(4, 2, dtype=torch.bool)
    changed_meta["episode_label_known_mask"][2:] = False
    second = routing.label_conditioned_episode(model, features, raw, changed_meta)
    for key, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[key], atol=0, rtol=0, msg=key)
    changed_features, changed_raw = features.clone(), raw.clone()
    changed_features[2:] *= -3
    changed_raw[2:] *= -3
    routing.label_conditioned_episode(model, changed_features, changed_raw, changed_meta)
    torch.testing.assert_close(conditions[1], conditions[3], atol=0, rtol=0)
    torch.testing.assert_close(conditions[1], conditions[5], atol=0, rtol=0)
    changed_features[1] *= -7
    changed_raw[1] *= -7
    routing.label_conditioned_episode(model, changed_features, changed_raw, changed_meta)
    torch.testing.assert_close(conditions[1][0], conditions[7][0], atol=0, rtol=0)


@pytest.mark.parametrize("empty", ["query", "support", "positive"])
def test_post_readout_empty_masks_and_no_positive_fallback(empty, monkeypatch):
    model = _new_model(classification=True, post_readout=True, checkpoint=True)
    model.label_pre_temporal.class_chunk_size = 2
    features, raw, meta = _episode()
    if empty == "positive":
        meta["episode_label_known_mask"] = torch.ones(4, 2, dtype=torch.bool)
        meta["episode_label_known_mask"][:2] = False
    else:
        sl = slice(2, None) if empty == "query" else slice(None, 2)
        meta["pred_query_mask"][sl] = False
        meta["pred_visibility"][sl] = False
    conditions = []
    fusion = model._fuse_episode_text_with_support_visual

    def record(*args):
        result = fusion(*args)
        conditions.append(result[0])
        return result

    monkeypatch.setattr(model, "_fuse_episode_text_with_support_visual", record)
    output = routing.label_conditioned_episode(model, features, raw, meta)
    if empty != "query":
        assert output["query_partial_support_prototypes"].eq(0).all()
        torch.testing.assert_close(conditions[1], torch.eye(4)[:2])
    loss = _loss(model, output, meta)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("detach", [False, True])
def test_post_condition_visual_detach_does_not_detach_matching_prototype(detach, monkeypatch):
    model = _new_model(post_readout=True)
    model.support_text_fusion_cfg.VISUAL_DETACH = detach
    features, raw, meta = _episode()
    features.requires_grad_()
    # Retain diagnostics only for this gradient oracle, not in production DDP.
    monkeypatch.setattr(routing, "_detach_diagnostics", lambda result: result)
    output = routing.label_conditioned_episode(model, features, raw, meta)
    gradient = torch.autograd.grad(output["query_partial_query_prototypes"].square().sum(),
                                   features, retain_graph=True)[0]
    assert (gradient[:2].abs().sum().item() == 0) == detach
    assert gradient[2:].abs().sum() > 0
    gradient = torch.autograd.grad(output["query_partial_q2s_base_logits"].sum(), features)[0]
    assert gradient[:2].abs().sum() > 0 and gradient[2:].abs().sum() > 0


@pytest.mark.parametrize("checkpoint,selective", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("class_chunk", [2, 4])
@pytest.mark.parametrize("detach", [False, True])
def test_post_readout_chunk_checkpoint_output_and_gradient_equivalence(checkpoint, selective, class_chunk, detach):
    a = _new_model(classification=True, post_readout=True)
    a.support_text_fusion_cfg.VISUAL_DETACH = detach
    a.head = nn.Linear(4, 3)
    a.cfg.TEST.SEEN_LABELS = [0, 1, 2]
    b = copy.deepcopy(a)
    b.label_pre_temporal.class_chunk_size = class_chunk
    b.label_pre_temporal.video_chunk_size = 1
    b.label_pre_temporal.batch_raw_quality = True
    b.label_pre_temporal.activation_checkpoint = checkpoint
    b.query_evidence_verifier.activation_checkpoint = checkpoint
    b.blocks[0].checkpoint_mlp = selective
    b.label_pre_temporal.readout_checkpoint = selective
    features, raw, meta = _episode()
    meta["episode_class_ids"] = torch.arange(3)
    meta["episode_positive_labels"] = torch.cat((meta["episode_positive_labels"],
                                               meta["episode_positive_labels"][:, 1:]), 1)
    x, y = features.clone().requires_grad_(), features.clone().requires_grad_()
    calls = []
    hook = b.blocks[0].temporal_attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    try:
        first = routing.label_conditioned_episode(a, x, raw, meta)
        second = routing.label_conditioned_episode(b, y, raw, meta)
        for key, value in first.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, second[key], atol=2e-5, rtol=2e-5, msg=key)
        _loss(a, first, meta).backward()
        _loss(b, second, meta).backward()
    finally:
        hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5, msg=name)


def test_v6_to_v8_explicit_init_and_v8_resume(tmp_path):
    old, model = _new_model(classification=True), _new_model(classification=True, post_readout=True)
    state = old.state_dict()
    with pytest.raises(ValueError, match="CHECKPOINT_EPOCH_RESET"):
        validate_label_pre_temporal_checkpoint(model, state)
    incomplete = dict(state)
    del incomplete["blocks.0.temporal_attn.qkv.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_label_pre_temporal_checkpoint(model, incomplete, allow_init=True)
    path = tmp_path / "v6.pyth"
    torch.save(dict(model_state=state, epoch=12, optimizer_state={}, scaler_state={}), path)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    assert load_checkpoint(str(path), model, data_parallel=False, optimizer=optimizer,
                           allow_label_pre_temporal_init=True) == -1
    assert model.label_pre_temporal.architecture_version.item() == 8 and not optimizer.state
    features, raw, meta = _episode()
    _loss(model, routing.label_conditioned_episode(model, features, raw, meta), meta).backward()
    optimizer.step()
    path = tmp_path / "v8.pyth"
    torch.save(dict(model_state=model.state_dict(), epoch=2, optimizer_state=optimizer.state_dict()), path)
    restored = _new_model(classification=True, post_readout=True)
    opt = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-3)
    assert load_checkpoint(str(path), restored, data_parallel=False, optimizer=opt) == 2
    assert restored.label_pre_temporal.architecture_version.item() == 8
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert opt.state[restored.head.weight]["step"] == optimizer.state[model.head.weight]["step"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_post_readout_cuda_amp_gradients_and_update(dtype, checkpoint, selective):
    model = _new_model(classification=True, post_readout=True, checkpoint=checkpoint).cuda()
    model.label_pre_temporal.class_chunk_size = 2
    model.label_pre_temporal.batch_raw_quality = True
    model.blocks[0].checkpoint_mlp = selective
    model.label_pre_temporal.readout_checkpoint = selective
    features, raw, meta = _episode("cuda")
    meta["pred_query_mask"][2, 1] = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):
        result = routing.label_conditioned_episode(model, features, raw, meta)
        loss = _loss(model, result, meta)
    before = model.head.weight.detach().clone()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.head.weight)


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_post_readout_two_rank_ddp_accumulation(tmp_path, cuda, checkpoint, selective):
    if cuda and (torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available()):
        pytest.skip("Two visible CUDA GPUs and NCCL required")
    if not cuda and not torch.distributed.is_gloo_available():
        pytest.skip("Gloo required")
    torch.multiprocessing.spawn(
        _single_path_ddp_worker,
        args=((tmp_path / "post_readout_rendezvous").as_uri(), cuda, 2, checkpoint, selective, selective, True, True),
        nprocs=2, join=True)
