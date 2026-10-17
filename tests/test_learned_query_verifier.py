"""Learned verifier boundaries, training integration and checkpoint safety."""

from pathlib import Path
from datetime import timedelta

import pytest
import torch
from torch import nn

from trokens.config.defaults import get_cfg
from trokens.config.query_judgments import resolve_query_judgments
from trokens.models.pointformer import Pointformer
from trokens.models.query_evidence_verifier import (
    QueryEvidenceVerifier, set_verifier_training_progress,
)
from tools.few_shot_multilabel import (
    compute_learned_evidence_mil_loss, compute_query_partial_q2s_loss,
    get_query_matchability_metrics,
)


def _cfg():
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs/trokens/sav.yaml"))
    return cfg


def _model(quality=True, uniqueness=True, checkpoint=False):
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = _cfg()
    cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    cfg.QUALITY_ENABLE = quality
    cfg.UNIQUENESS_ENABLE = uniqueness
    vc = cfg.LEARNED_VERIFIER
    vc.QUALITY_HIDDEN_DIM = 8
    vc.QUERY_CHUNK_SIZE = 1
    vc.ACTIVATION_CHECKPOINT = checkpoint
    model.embed_dim = 4
    model.pot_route_cfg = model.cfg.FEW_SHOT.POT_ROUTE
    model.pot_route_cfg.FRAME_SOFTMAX_TAU = 0.4
    model.use_frame_softmax_route = model.use_query_partial_q2s = True
    model.use_support_text_fusion = True
    model.support_text_fusion_cfg = model.cfg.FEW_SHOT.SUPPORT_TEXT_FUSION
    model._get_pot_label_text_features = lambda ids, dtype: torch.eye(4, device=ids.device, dtype=dtype)[:2]
    model._init_query_evidence_verifier()
    return model


def _episode(device="cpu"):
    generator = torch.Generator().manual_seed(414)
    post = torch.randn(4, 3, 5, 4, generator=generator).to(device)
    raw = torch.randn(4, 3, 5, 4, generator=generator).to(device)
    mask = torch.ones(4, 3, 5, dtype=torch.bool, device=device)
    visibility = mask.clone()
    visibility[0, 1, 2:] = False
    visibility[2, 1] = False  # invalid Query frame
    metadata = {
        "support_mask": torch.tensor([True, True, False, False], device=device),
        "pred_query_mask": mask,
        "pred_visibility": visibility,
        "episode_class_ids": torch.tensor([0, 1], device=device),
        "episode_positive_labels": torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], device=device).bool(),
    }
    return post, raw, metadata


def _run(model, post, raw, meta):
    return model._build_frame_softmax_q2s_aux(post, meta, matchability_evidence_tokens=raw)


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("uniqueness", [False, True])
def test_switches_preserve_route_and_independent_head_gradients(quality, uniqueness, monkeypatch):
    model = _model(quality, uniqueness)
    post, raw, meta = _episode()
    post.requires_grad_()
    out = _run(model, post, raw if quality else None, meta)
    base_model = _model(False, False)
    base = _run(base_model, post, None, meta)
    torch.testing.assert_close(out["query_partial_query_prototypes"], base["query_partial_query_prototypes"], rtol=0, atol=0)
    torch.testing.assert_close(out["query_partial_q2s_base_logits"], base["query_partial_q2s_base_logits"], rtol=0, atol=0)
    assert ("query_quality_logits" in out) == quality
    assert ("query_relation_logits" in out) == uniqueness
    assert "query_frame_relative_margin" not in out
    assert "query_absolute_mass_threshold" not in out
    if quality or uniqueness:
        m = out.get("query_frame_absolute_patch_mass", 1.0)
        rho = out.get("query_frame_matchability", 1.0)
        torch.testing.assert_close(out["query_frame_target_mass"], m * rho)
        total = sum(out[f"query_frame_{state}_mass"] for state in ("target", "confuser", "null"))
        torch.testing.assert_close(total, torch.ones_like(total))
        assert torch.all(out["query_partial_q2s_logits"] <= out["query_partial_q2s_base_logits"] + 1e-6)
    else:
        assert torch.equal(out["query_partial_q2s_logits"], out["query_partial_q2s_base_logits"])
    labels = meta["episode_positive_labels"][2:].float()
    loss, _, metrics = compute_query_partial_q2s_loss(out["query_partial_q2s_logits"], labels, out, model.cfg)
    assert ("q2s_quality_mil_loss" in metrics) == quality
    assert ("q2s_relation_mil_loss" in metrics) == uniqueness
    assert "q2s_evidence_mil_loss" not in metrics
    loss.backward()
    assert post.grad is not None and torch.isfinite(post.grad).all()
    if quality or uniqueness:
        for name, parameter in model.query_evidence_verifier.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    reported = get_query_matchability_metrics(out, labels)
    assert not any("margin" in name for name in reported)


def test_query_labels_ignored_and_support_order_invariant():
    model = _model().eval()
    post, raw, meta = _episode()
    first = _run(model, post, raw, meta)
    changed = dict(meta)
    changed["episode_positive_labels"] = meta["episode_positive_labels"].clone()
    changed["episode_positive_labels"][2:] = ~changed["episode_positive_labels"][2:]
    second = _run(model, post, raw, changed)
    order = torch.tensor([1, 0, 2, 3])
    permuted = {k: (v[order] if v.ndim and v.shape[0] == 4 else v) for k, v in meta.items()}
    third = _run(model, post[order], raw[order], permuted)
    for key in ("query_quality_logits", "query_relation_logits", "query_partial_q2s_logits"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(first[key], third[key], atol=2e-6, rtol=2e-6)


def test_heads_share_parameters_across_episode_classes():
    verifier = _model().query_evidence_verifier.eval()
    inputs = _direct_inputs()
    first = verifier(**inputs)
    order = torch.tensor([1, 0])
    permuted = dict(inputs, text=inputs["text"][order],
                    post_references=inputs["post_references"][:, order],
                    support_frame_valid=inputs["support_frame_valid"][:, order],
                    query_weights=inputs["query_weights"][:, order],
                    support_labels=inputs["support_labels"][:, order])
    second = verifier(**permuted)
    for key in ("quality_logits", "relation_logits"):
        # Batched GEMM may change FP32 reduction ordering after a permutation.
        torch.testing.assert_close(first[key][:, order], second[key], rtol=2e-6, atol=2e-7)


def test_missing_references_invalid_frames_and_unknown_support_are_safe():
    model = _model()
    post, raw, meta = _episode()
    meta["episode_positive_labels"][:2] = True  # no negative Support
    out = _run(model, post, raw, meta)
    valid = meta["pred_visibility"][2:].any(-1)[:, None].expand(2, 2, 3)
    torch.testing.assert_close(out["query_frame_absolute_patch_mass"], valid.float())
    assert torch.equal(out["query_frame_matchability"], torch.ones(2, 2, 3))
    assert not out["query_quality_valid"].any()
    assert not out["query_relation_valid"].any()
    loss, _, _ = compute_query_partial_q2s_loss(out["query_partial_q2s_logits"], torch.ones(2, 2), out, model.cfg)
    loss.backward()
    for name, p in model.query_evidence_verifier.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    meta["episode_positive_labels"][:2] = torch.tensor([[1, 0], [0, 1]]).bool()
    meta["episode_label_known_mask"] = torch.ones_like(meta["episode_positive_labels"])
    meta["episode_label_known_mask"][0, 1] = False
    meta["episode_label_known_mask"][1, 0] = False
    out = _run(model, post, raw, meta)
    assert not out["query_relation_reference_available"].any()
    assert not out["query_quality_reference_available"].any()
    del meta["episode_label_known_mask"]
    meta["episode_positive_labels"] = meta["episode_positive_labels"].float()
    meta["episode_positive_labels"][0, 1] = -1
    meta["episode_positive_labels"][1, 0] = float("nan")
    out = _run(model, post, raw, meta)
    assert not out["query_relation_reference_available"].any()
    assert not out["query_quality_reference_available"].any()


def _direct_inputs(device="cpu"):
    torch.manual_seed(82)
    shape = (2, 3, 4, 4)
    values = lambda: torch.randn(shape, device=device, requires_grad=True)
    return dict(
        raw_query=values(), raw_support=values(), text=torch.randn(2, 4, device=device, requires_grad=True),
        post_query=values(), post_references=torch.randn(2, 2, 3, 4, device=device, requires_grad=True),
        query_weights=torch.full((2, 2, 3, 4), 0.25, device=device, requires_grad=True),
        query_valid=torch.ones(shape[:3], device=device, dtype=torch.bool),
        support_raw_valid=torch.ones(shape[:3], device=device, dtype=torch.bool),
        support_frame_valid=torch.ones(2, 2, 3, device=device, dtype=torch.bool),
        support_labels=torch.eye(2, device=device).bool(),
    )


def test_shared_relation_swap_symmetry_and_detached_inputs():
    verifier = _model().query_evidence_verifier
    inputs = _direct_inputs()
    first = verifier(**inputs)
    swapped = dict(inputs, support_labels=~inputs["support_labels"])
    second = verifier(**swapped)
    torch.testing.assert_close(first["relation"], 1 - second["relation"], atol=1e-7, rtol=1e-6)
    same = dict(inputs, post_references=inputs["post_references"][0:1].expand(2, -1, -1, -1))
    # Equal sets occupy different masked positions; FP32 logsumexp reduction
    # order can differ by one ulp in metric mode.
    torch.testing.assert_close(verifier(**same)["relation"], torch.full((2, 2, 3), 0.5), rtol=0, atol=1e-7)
    (first["quality_logits"].square().mean() + first["relation_logits"].square().mean()).backward()
    for name, value in inputs.items():
        if value.requires_grad:
            assert value.grad is None, name
    for name, p in verifier.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_chunking_checkpoint_and_autocast_finite_gradients(use_checkpoint):
    verifier = _model(checkpoint=use_checkpoint).query_evidence_verifier
    inputs = _direct_inputs()
    inputs["query_valid"][0, 1] = False
    inputs["support_raw_valid"][0, 0] = False
    first = verifier(**inputs)
    verifier.chunk_size = 8
    with torch.autocast("cpu", dtype=torch.bfloat16):
        second = verifier(**inputs)
        loss = sum(second[key].square().mean() for key in ("quality_logits", "relation_logits"))
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in verifier.parameters())


def test_new_mil_unknown_labels_empty_bags_and_nonwinning_gradients():
    logits = torch.tensor([[[0.1, 0.5, -0.2], [1., 2., 3.]], [[0., 1., 2.], [2., 3., 4.]]], requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[0, 1] = False
    labels = torch.tensor([[1., 0.], [-1., 0.]])
    out = compute_learned_evidence_mil_loss(logits, valid, labels)
    assert out["valid"].tolist() == [[True, False], [False, True]]
    expected = torch.logsumexp(logits[0, 0], dim=0) - torch.tensor(3.).log()
    torch.testing.assert_close(out["logits"][0, 0], expected)
    out["loss"].backward()
    assert logits.grad[0, 0].abs().min() > 0
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[1, 0].abs().sum() == 0
    assert torch.isfinite(logits.grad).all()


def test_warmup_is_training_only_and_resumable():
    model = _model()
    post, raw, meta = _episode()
    set_verifier_training_progress(model, 0, 100)
    first = _run(model, post, raw, meta)
    assert torch.equal(first["query_partial_q2s_logits"], first["query_partial_q2s_base_logits"])
    set_verifier_training_progress(model, 3, 100)
    assert model.query_evidence_verifier.transport_progress == pytest.approx(0.6)
    assert _run(model.eval(), post, raw, meta)["query_verifier_transport_strength"].item() == 1.0


def test_independent_switches_and_tinyvirat_uses_current_verifier():
    cfg = _cfg()
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert resolve_query_judgments(match).quality
    assert resolve_query_judgments(match).uniqueness
    match.QUALITY_ENABLE = False
    assert not resolve_query_judgments(match).quality
    assert resolve_query_judgments(match).uniqueness
    tiny = get_cfg()
    tiny.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs/trokens/tinyvirat.yaml"))
    assert tiny.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE
    vc = tiny.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    assert "QUALITY_MODE" not in vc and "UNIQUENESS_MODE" not in vc


def test_verified_loss_trains_heads_and_matcher_without_raw_feedback():
    model = _model()
    post, raw, meta = _episode()
    post.requires_grad_()
    raw.requires_grad_()
    out = _run(model, post, raw, meta)
    torch.nn.functional.binary_cross_entropy_with_logits(
        out["query_partial_q2s_logits"], meta["episode_positive_labels"][2:].float(),
    ).backward()
    assert raw.grad is None
    assert post.grad is not None and post.grad.abs().sum() > 0
    verifier = model.query_evidence_verifier
    quality_weight = verifier.semantic_quality.relation[-1].weight
    for parameter in (quality_weight, verifier.metric_temperature_raw, verifier.metric_scale_raw):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("quality", [False, True])
def test_forward_extracts_raw_before_position_and_motion_only_when_needed(quality):
    model = _model(quality=quality)
    _, raw, meta = _episode()
    meta["pred_tracks"] = torch.zeros(4, 3, 5, 2)
    meta["hod_feat"] = torch.zeros_like(raw)
    model.feat_extractor_type = "dinotxt_vitl14_reg4"
    model.cfg.POINT_INFO.ENABLE = True
    model.cfg.MF.USE_BASE_POS_EMBED = True
    model.head_drop = model.head = nn.Identity()
    model.get_dinotxt_features = lambda video: raw.unsqueeze(2)
    sampled = []
    def sample(feature, tracks, add_pt_pos_embed):
        sampled.append(feature)
        return feature.flatten(2, 3)
    model._sample_point_features = sample
    model.add_st_pos_embeddings = lambda feature: feature + 10
    model.hod_motion_module = lambda feature: torch.ones_like(raw)
    model.cross_motion_module = lambda tracks, visibility: torch.full_like(raw, 2)
    model.pt_forward = lambda feature, metadata: (feature.mean((1, 2)), feature + 4)
    captured = {}
    original = model.query_evidence_verifier.forward
    def verify(**inputs):
        captured.update(inputs)
        return original(**inputs)
    model.query_evidence_verifier.forward = verify
    _, post, aux = model({"video": torch.empty(4, 3, 3, 2, 2), "metadata": meta})
    assert len(sampled) == (2 if quality else 1)
    torch.testing.assert_close(post, raw + 17)
    torch.testing.assert_close(captured["post_query"], post[2:])
    if quality:
        torch.testing.assert_close(captured["raw_query"], raw[2:], rtol=0, atol=0)
        torch.testing.assert_close(captured["raw_support"], raw[:2], rtol=0, atol=0)
    else:
        assert captured["raw_query"] is None
    assert torch.isfinite(aux["query_partial_q2s_logits"]).all()


def test_checkpoint_guard_roundtrip_and_optimizer_group(tmp_path):
    from trokens.models.optimizer import construct_optimizer
    from trokens.utils.checkpoint import load_checkpoint, validate_verifier_checkpoint
    model = _model()
    opt = construct_optimizer(model, model.cfg)
    grouped = {id(p): group["weight_decay"] for group in opt.param_groups for p in group["params"]}
    for name, p in model.query_evidence_verifier.named_parameters():
        assert grouped[id(p)] == (0.0 if p.ndim <= 1 else 0.01), name
    state = model.state_dict()
    assert validate_verifier_checkpoint(model, state) is False
    with pytest.raises(ValueError, match="lacks complete"):
        validate_verifier_checkpoint(model, {})
    assert validate_verifier_checkpoint(model, {}, allow_init=True)
    partial = {"query_evidence_verifier.architecture_version": state["query_evidence_verifier.architecture_version"]}
    with pytest.raises(ValueError, match="lacks complete"):
        validate_verifier_checkpoint(model, partial, allow_init=True)
    path = tmp_path / "new.pyth"
    torch.save({"model_state": state, "epoch": 4, "optimizer_state": opt.state_dict()}, path)
    restored = _model()
    assert load_checkpoint(str(path), restored, data_parallel=False) == 4
    for key in state:
        assert torch.equal(state[key], restored.state_dict()[key])
    old = tmp_path / "old.pyth"
    torch.save({"model_state": {}, "epoch": 19, "optimizer_state": {}}, old)
    with pytest.raises(ValueError, match="lacks complete"):
        load_checkpoint(str(old), restored, data_parallel=False)
    assert load_checkpoint(str(old), restored, data_parallel=False, optimizer=opt,
                           allow_verifier_init=True) == -1
    incompatible = dict(state)
    projection = "semantic_quality.projection"
    incompatible[f"query_evidence_verifier.{projection}.weight"] = torch.zeros(7, 4)
    bad = tmp_path / "incompatible.pyth"
    torch.save({"model_state": incompatible, "epoch": 4}, bad)
    for inflation in (False, True):
        with pytest.raises(ValueError, match="lacks complete"):
            load_checkpoint(str(bad), restored, data_parallel=False,
                            allow_verifier_init=True, inflation=inflation)


def test_real_registration_preserves_backbone_rng_and_weights(monkeypatch):
    monkeypatch.setattr(Pointformer, "_load_dinotxt_visual_model", lambda self: nn.Identity())
    monkeypatch.setattr(Pointformer, "_load_dinotxt_text_model", lambda self: nn.Identity())
    monkeypatch.setattr(Pointformer, "_load_dinotxt_tokenizer", lambda self: None)
    cfg = _cfg()
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = False
    torch.manual_seed(59)
    baseline = Pointformer(cfg)
    state, rng = baseline.state_dict(), torch.random.get_rng_state()
    cfg = cfg.clone()
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = True
    torch.manual_seed(59)
    learned = Pointformer(cfg)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in state.items():
        assert torch.equal(value, learned.state_dict()[key]), key
    assert learned.query_evidence_verifier is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_cuda_amp_partial_frames_backward(checkpoint):
    model = _model(checkpoint=checkpoint).cuda()
    model.query_evidence_verifier.chunk_size = 16
    post, raw, meta = _episode("cuda")
    post.requires_grad_()
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with torch.autocast("cuda", dtype=torch.float16):
        out = _run(model, post, raw, meta)
        loss, _, _ = compute_query_partial_q2s_loss(
            out["query_partial_q2s_logits"], meta["episode_positive_labels"][2:].float(), out, model.cfg,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss) and torch.isfinite(post.grad).all()
    for key in ("query_frame_target_mass", "query_frame_confuser_mass", "query_frame_null_mass"):
        assert out[key].dtype == torch.float32
    torch.testing.assert_close(out["query_frame_target_mass"],
                               out["query_frame_absolute_patch_mass"] * out["query_frame_matchability"])
    for name, p in model.query_evidence_verifier.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        # Match production DDP + activation checkpoint settings. One worker
        # sometimes has no valid supervision, but all heads must stay connected.
        verifier = DistributedDataParallel(_model(checkpoint=True).query_evidence_verifier,
                                           find_unused_parameters=True)
        optimizer = torch.optim.SGD(verifier.parameters(), lr=0.01)
        for step in range(3):
            inputs = _direct_inputs()
            if rank == 1 and step == 0:
                inputs["support_labels"][:] = True  # no Confusers on this rank
            if rank == 1 and step == 1:
                inputs["query_valid"][:] = False
            output = verifier(**inputs)
            loss = sum(compute_learned_evidence_mil_loss(
                output[name + "_logits"], output[name + "_valid"], torch.eye(2),
            )["loss"] for name in ("quality", "relation"))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in verifier.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            optimizer.step()
        flat = torch.cat([p.detach().flatten() for p in verifier.parameters()])
        replicas = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(replicas, flat)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_two_rank_ddp_checkpointing_with_missing_references(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=((tmp_path / "rendezvous").as_uri(),),
                                nprocs=2, join=True)
