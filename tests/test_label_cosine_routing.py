"""Parameter-free early cosine routing; preserve trajectory encoder and losses."""

import copy
from pathlib import Path

import pytest
import torch

from test_label_pre_temporal import _new_model, _episode, _loss, _single_path_ddp_worker
from test_label_asymmetric_conditions import _capture_conditions
from trokens.config.defaults import get_cfg
from trokens.models.label_conditioned_routing import label_conditioned_episode
from trokens.utils.checkpoint import validate_label_pre_temporal_checkpoint, load_checkpoint


def _cosine_model(**kwargs):
    return _new_model(scoring="cosine", **kwargs)


def test_cosine_config_changes_only_scoring_and_output():
    configs = []
    for name in ("sav_label_pre_temporal_post_readout.yaml", "sav_label_pre_temporal_cosine.yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs/trokens" / name))
        configs.append(cfg)
    old, new = configs
    assert old.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING == "projected_cosine"
    assert new.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING == "cosine"
    assert "POST_SUPPORT_READOUT" not in new.FEW_SHOT.LABEL_PRE_TEMPORAL
    assert new.OUTPUT_DIR != old.OUTPUT_DIR
    new.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING = "projected_cosine"
    new.OUTPUT_DIR = old.OUTPUT_DIR
    assert new == old
    with pytest.raises(ValueError, match="SCORING"):
        _new_model(scoring="dot_product")


@pytest.mark.parametrize("per_video", [False, True])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_direct_cosine_weights_match_legacy_formula_with_empty_and_zero_rows(per_video, nonfinite):
    model = _cosine_model()
    router = model.label_pre_temporal
    _, raw, meta = _episode()
    mask = meta["pred_query_mask"].clone()
    mask[0, 0] = False
    mask[1, :, 1] = False
    raw[0, 1] = 0
    text = torch.randn(4, 3, 4) if per_video else torch.randn(3, 4)
    if nonfinite:
        raw[2, 1, 2, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        text.reshape(-1, 4)[0, 0] = float("nan")
    actual = router.routing_weights(raw, text, mask)
    expected = []
    for b in range(4):
        # Use the actual legacy cosine + masked-softmax implementation as an
        # independent oracle. Its AV output is NOT used by the early encoder.
        labels = text[b] if per_video else text
        expected.append(model._compute_batched_frame_softmax_text_prototypes(
            raw[b:b+1], mask[b:b+1], labels)[1])
    expected = torch.cat(expected)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected_gate = 1 + router.strength * (mask.sum(-1)[:, None, :, None] * expected - 1).tanh()
    expected_gate = torch.where(mask[:, None], expected_gate, torch.ones_like(expected_gate))
    gate = router(raw, text, mask)
    torch.testing.assert_close(gate, expected_gate, atol=2e-6, rtol=2e-6)
    assert actual[0, :, 0].eq(0).all() and gate[0, :, 0].eq(1).all()
    # No class-axis competition: duplicating a label does not split its mass.
    duplicate = text[:, :1].expand(-1, 2, -1) if per_video else text[:1].expand(2, -1)
    weights = router.routing_weights(raw, duplicate, mask)
    torch.testing.assert_close(weights[:, 0], weights[:, 1])
    assert not list(router.parameters())
    assert set(router.state_dict()) == {"architecture_version"}
    assert any(p.requires_grad for p in model.blocks[0].temporal_attn.qkv.parameters())
    assert any(p.requires_grad for p in model.blocks[0].attn.qkv.parameters())


def test_original_space_cosine_preserves_raw_and_condition_gradients():
    model = _cosine_model()
    _, raw, meta = _episode()
    raw.requires_grad_()
    text = torch.randn(2, 4, requires_grad=True)
    actual = model.label_pre_temporal.routing_weights(raw, text, meta["pred_query_mask"])
    expected = model._compute_batched_frame_softmax_text_prototypes(raw, meta["pred_query_mask"], text)[1]
    direction = torch.randn_like(actual)
    a = torch.autograd.grad((actual * direction).sum(), (raw, text), retain_graph=True)
    b = torch.autograd.grad((expected * direction).sum(), (raw, text))
    for x, y in zip(a, b):
        assert torch.isfinite(x).all() and x.abs().sum() > 0
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)


def test_initial_episode_matches_identity_projected_control_and_shared_gradients():
    control = _new_model(classification=True)
    model = _cosine_model(classification=True)
    # The constructor preserves all shared initial weights and RNG state.
    for name, value in model.state_dict().items():
        if name != "label_pre_temporal.architecture_version":
            torch.testing.assert_close(value, control.state_dict()[name], atol=0, rtol=0)
    features, raw, meta = _episode()
    x, y = features.clone().requires_grad_(), features.clone().requires_grad_()
    calls = []
    hook = model.blocks[0].temporal_attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    try:
        a = label_conditioned_episode(control, x, raw, meta)
        b = label_conditioned_episode(model, y, raw, meta)
        for key, value in a.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, b[key], atol=2e-6, rtol=2e-6, msg=key)
        _loss(control, a, meta).backward()
        _loss(model, b, meta).backward()
    finally:
        hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-6)
    reference = dict(control.named_parameters())
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, reference[name].grad, atol=2e-6, rtol=2e-6, msg=name)
    # Unlike learned Wq/Wk, direct-cosine weights cannot drift after updates.
    before = model.label_pre_temporal(raw, torch.eye(4)[:2], meta["pred_query_mask"])
    torch.optim.SGD(model.parameters(), lr=.01).step()
    after = model.label_pre_temporal(raw, torch.eye(4)[:2], meta["pred_query_mask"])
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_direct_route_is_still_pre_encoding_with_asymmetric_conditions_and_query_isolation():
    model = _cosine_model(classification=True).eval()
    features, raw, meta = _episode()
    before, conditions, gates = _capture_conditions(model, features, raw, meta)
    text = torch.eye(4)[:2]
    torch.testing.assert_close(conditions[:2], text.expand(2, -1, -1))
    assert not torch.allclose(conditions[2:], conditions[:2])
    changed = copy.deepcopy(meta)
    changed["episode_positive_labels"] = changed["episode_positive_labels"].float()
    changed["episode_positive_labels"][2:] = float("nan")
    after, next_conditions, next_gates = _capture_conditions(model, features, raw, changed)
    for key, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, after[key], atol=0, rtol=0)
    torch.testing.assert_close(conditions, next_conditions, atol=0, rtol=0)
    torch.testing.assert_close(gates, next_gates, atol=0, rtol=0)
    for name in ("temporal_bias_scale", "spatial_bias_scale"):
        setattr(model.label_pre_temporal, name, 0.)
        ablated = label_conditioned_episode(model, features, raw, meta)
        assert not torch.allclose(before["global_classification_logits"],
                                  ablated["global_classification_logits"])
        setattr(model.label_pre_temporal, name, 1.)


@pytest.mark.parametrize("quality,uniqueness", [(True, True), (False, False)])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
@pytest.mark.parametrize("detach", [False, True])
def test_cosine_chunk_checkpoint_empty_masks_and_gradients(quality, uniqueness, checkpoint, selective, detach):
    a = _cosine_model(classification=True, quality=quality, uniqueness=uniqueness)
    a.support_text_fusion_cfg.VISUAL_DETACH = detach
    b = copy.deepcopy(a)
    b.label_pre_temporal.class_chunk_size = 2
    b.label_pre_temporal.video_chunk_size = 1
    b.label_pre_temporal.batch_raw_quality = True
    b.label_pre_temporal.activation_checkpoint = checkpoint
    b.blocks[0].checkpoint_mlp = selective
    b.label_pre_temporal.readout_checkpoint = selective
    if b.query_evidence_verifier is not None:
        b.query_evidence_verifier.activation_checkpoint = checkpoint
    features, raw, meta = _episode()
    # Last class group is incomplete and has no positive Support reference.
    meta["episode_class_ids"] = torch.arange(3)
    meta["episode_positive_labels"] = torch.cat((meta["episode_positive_labels"],
                                               torch.zeros(4, 1, dtype=torch.bool)), 1)
    meta["pred_query_mask"][0, 1] = False
    meta["pred_query_mask"][2, 1] = False
    a.head = torch.nn.Linear(4, 3)
    b.head = copy.deepcopy(a.head)
    x, y = features.clone().requires_grad_(), features.clone().requires_grad_()
    first = label_conditioned_episode(a, x, raw, meta)
    second = label_conditioned_episode(b, y, raw, meta)
    for key, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[key], atol=3e-5, rtol=3e-5, msg=key)
    _loss(a, first, meta).backward()
    _loss(b, second, meta).backward()
    assert x.grad[:2].abs().sum() > 0 and x.grad[2:].abs().sum() > 0
    torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5, msg=name)


@pytest.mark.parametrize("classification", [False, True])
def test_cosine_checkpoint_semantics_and_cross_scorer_rejection(classification):
    model = _new_model(scoring="cosine", classification=classification)
    assert model.label_pre_temporal.architecture_version.item() == 31 + int(classification)
    assert not validate_label_pre_temporal_checkpoint(model, model.state_dict())
    learned = _new_model(classification=classification)
    for destination, source in ((model, learned), (learned, model)):
        for allow_init in (False, True):
            with pytest.raises(ValueError, match="SCORING mismatch"):
                validate_label_pre_temporal_checkpoint(destination, source.state_dict(), allow_init=allow_init)
    for version in (9, 10, 11, 12, 25, 26, 27, 28, 31, 32):
        if version == model.label_pre_temporal.architecture_version.item():
            continue
        state = model.state_dict()
        state["label_pre_temporal.architecture_version"] = torch.tensor(version)
        with pytest.raises(ValueError, match="CHECKPOINT_EPOCH_RESET"):
            validate_label_pre_temporal_checkpoint(model, state)
        assert validate_label_pre_temporal_checkpoint(model, state, allow_init=True)
    state = model.state_dict()
    del state["blocks.0.temporal_attn.qkv.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_label_pre_temporal_checkpoint(model, state)


def test_cosine_checkpoint_restores_optimizer_and_predictions(tmp_path):
    model = _cosine_model(classification=True)
    features, raw, meta = _episode()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    _loss(model, label_conditioned_episode(model, features, raw, meta), meta).backward()
    optimizer.step()
    path = tmp_path / "cosine.pyth"
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=4), path)
    restored = _cosine_model(classification=True)
    opt = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-4)
    assert load_checkpoint(str(path), restored, data_parallel=False, optimizer=opt) == 4
    assert restored.label_pre_temporal.architecture_version.item() == 32
    assert opt.state[restored.head.weight]["step"] == optimizer.state[model.head.weight]["step"]
    first = label_conditioned_episode(model.eval(), features, raw, meta)
    second = label_conditioned_episode(restored.eval(), features, raw, meta)
    for key, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[key], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_cosine_cuda_amp_keeps_scorer_fp32_and_trains_encoder(dtype, checkpoint, selective):
    model = _cosine_model(classification=True, checkpoint=checkpoint).cuda()
    model.label_pre_temporal.class_chunk_size = 2
    model.blocks[0].checkpoint_mlp = selective
    model.label_pre_temporal.readout_checkpoint = selective
    features, raw, meta = _episode("cuda")
    text = torch.eye(4, device="cuda")[:2]
    expected = model.label_pre_temporal.routing_weights(raw, text, meta["pred_query_mask"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):
        actual = model.label_pre_temporal.routing_weights(raw, text, meta["pred_query_mask"])
        output = label_conditioned_episode(model, features, raw, meta)
        loss = _loss(model, output, meta)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.float32
    before = model.blocks[0].temporal_attn.qkv.weight.detach().clone()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.blocks[0].temporal_attn.qkv.weight)


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_cosine_two_rank_ddp_accumulation(tmp_path, cuda, checkpoint, selective):
    if cuda and (torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available()):
        pytest.skip("Two visible CUDA GPUs and NCCL required")
    if not cuda and not torch.distributed.is_gloo_available():
        pytest.skip("Gloo required")
    torch.multiprocessing.spawn(
        _single_path_ddp_worker,
        args=((tmp_path / "cosine_rendezvous").as_uri(), cuda, 2, checkpoint, selective, selective,
              True, "cosine"), nprocs=2, join=True)
