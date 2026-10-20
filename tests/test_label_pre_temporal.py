"""Single-path routing: serial equivalence, isolation, gradients and contracts."""

import copy
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import nn

from test_learned_query_verifier import _model, _episode, _run
from trokens.models.attention import TrajectoryAttentionBlock
from trokens.models.label_conditioned_routing import (
    LabelPreTemporalRouting, _prepare, _encode, label_conditioned_episode,
)
from trokens.utils.checkpoint import validate_label_pre_temporal_checkpoint, load_checkpoint
from tools.few_shot_multilabel import (
    compute_query_partial_q2s_loss, single_path_episode_targets, unpack_episode_output,
    multilabel_classification_loss,
)


def _new_model(quality=True, uniqueness=True, checkpoint=False, cls=True, classification=False,
               post_readout=False, scoring="projected_cosine"):
    torch.manual_seed(71)
    model = _model(quality, uniqueness, checkpoint)
    model.cfg.FEW_SHOT.CLASS_LOSS_LAMBDA = 1. if classification else 0.
    model.cfg.TEST.SEEN_LABELS = [0, 1]
    model.cfg.MODEL.USE_CLS_TOKEN = cls
    model.cfg.MF.PT_ATTENTION = "divided_space_time"
    model.cfg.MODEL.ADAPOOLING.ENABLE = False
    model.cfg.MODEL.TRAIN_BACKBONE = False
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE = True
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING = scoring
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ROUTE_TEMPERATURE = .4
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.ACTIVATION_CHECKPOINT = checkpoint
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT = post_readout
    model.depth = 1
    model.point_grid_size = 1
    model.temporal_resolution = 3
    model.use_label_pre_temporal = True
    model.cls_token = nn.Parameter(torch.randn(1, 1, 4)) if cls else nn.Identity()
    model.pos_drop = nn.Identity()
    model.blocks = nn.ModuleList([TrajectoryAttentionBlock(
        cfg=model.cfg, dim=4, num_heads=2, mlp_ratio=2., qkv_bias=True,
        use_pt_visibility=True)])
    model.norm = nn.LayerNorm(4)
    model.head = nn.Linear(4, 2)
    model.head_drop = nn.Identity()
    model.pre_logits = nn.Linear(4, 4)
    model._init_label_pre_temporal()
    model._get_pot_label_text_features = lambda ids, dtype: torch.eye(4, device=ids.device, dtype=dtype)[ids]
    return model


def _loss(model, output, metadata):
    q2s = compute_query_partial_q2s_loss(
        output["query_partial_q2s_logits"], metadata["episode_positive_labels"][2:].float(),
        output, model.cfg)[0]
    if not model.label_pre_temporal.global_classifier:
        return q2s
    cls = multilabel_classification_loss(
        output["global_classification_logits"] / model.cfg.SOLVER.TEMPRATURE,
        metadata["episode_positive_labels"].float(), model.cfg, nn.BCEWithLogitsLoss())
    return model.cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * cls + model.cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s


@pytest.mark.parametrize("cls", [False, True])
def test_neutral_gate_matches_original_serial_encoder_and_gradients(cls):
    model = _new_model(cls=cls).eval()
    features, _, meta = _episode()
    mask = meta["pred_query_mask"].clone()
    mask[0, :, 0] = False
    mask[2, 1] = False
    meta["pred_query_mask"] = mask
    features.requires_grad_()
    expected = model.pt_forward(features, meta)[1]
    tokens, token_mask, state = _prepare(model, features, mask)
    actual = _encode(model, tokens, token_mask, state, torch.ones_like(mask).float())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    params = (features, *model.blocks.parameters())
    a = torch.autograd.grad((actual * gradient).sum(), params, retain_graph=True)
    b = torch.autograd.grad((expected * gradient).sum(), params)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)


def test_route_uniform_empty_rows_bounds_and_nonexclusive_class_axis():
    model = _new_model()
    router = model.label_pre_temporal
    raw = torch.ones(2, 3, 5, 4)
    mask = torch.ones(2, 3, 5).bool()
    mask[0, 0] = False
    labels = torch.randn(2, 4)
    gate = router(raw, labels, mask)
    torch.testing.assert_close(gate, torch.ones_like(gate))
    raw = torch.randn_like(raw)
    gate = router(raw, labels[:1].expand(2, -1), mask)
    torch.testing.assert_close(gate[:, 0], gate[:, 1])
    assert (gate >= .75).all() and (gate <= 1.25).all()
    gate.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())


def test_fast_config_preserves_model_contract_and_selective_checkpoint_flags():
    from trokens.config.defaults import get_cfg
    base, fast = get_cfg(), get_cfg()
    root = Path(__file__).resolve().parents[1]
    base.merge_from_file(str(root / "configs/trokens/sav_label_pre_temporal.yaml"))
    fast.merge_from_file(str(root / "configs/trokens/sav_label_pre_temporal_fast.yaml"))
    assert fast.FEW_SHOT.LABEL_PRE_TEMPORAL.CLASS_CHUNK_SIZE == 5
    assert fast.FEW_SHOT.LABEL_PRE_TEMPORAL.VIDEO_CHUNK_SIZE == 8
    assert not fast.FEW_SHOT.LABEL_PRE_TEMPORAL.ACTIVATION_CHECKPOINT
    assert fast.FEW_SHOT.LABEL_PRE_TEMPORAL.MLP_CHECKPOINT
    assert fast.FEW_SHOT.LABEL_PRE_TEMPORAL.READOUT_CHECKPOINT
    assert not fast.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.ACTIVATION_CHECKPOINT
    assert base.FEW_SHOT.CLASS_LOSS_LAMBDA == fast.FEW_SHOT.CLASS_LOSS_LAMBDA == 1.
    for cfg in (base, fast):
        assert cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.TEMPORAL_BIAS_SCALE == 1.
        assert cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.SPATIAL_BIAS_SCALE == 1.
        assert cfg.OUTPUT_DIR.endswith("_st_bias_cls")
    for name in ("MODEL", "MF", "POINT_INFO", "SOLVER", "DATA"):
        assert base[name] == fast[name]
    model = _new_model(checkpoint=True)
    model.cfg.FEW_SHOT.LABEL_PRE_TEMPORAL.MLP_CHECKPOINT = True
    with pytest.raises(ValueError, match="selective"):
        model._init_label_pre_temporal()


@pytest.mark.parametrize("quality,uniqueness", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_episode_finite_loss_and_gradients_without_global_branch(quality, uniqueness, checkpoint):
    model = _new_model(quality, uniqueness, checkpoint)
    features, raw, meta = _episode()
    features.requires_grad_()
    output = label_conditioned_episode(model, features, raw, meta)
    assert output["single_path_episode"]
    assert output["query_partial_query_prototypes"].shape == (2, 2, 3, 4)
    assert all(not (torch.is_tensor(v) and v.ndim == 5) for v in output.values())
    loss = _loss(model, output, meta)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert features.grad[:2].abs().sum() > 0  # Support matcher is not detached.
    assert features.grad[2:].abs().sum() > 0
    assert model.label_pre_temporal.query.weight.grad.abs().sum() > 0
    assert model.label_pre_temporal.key.weight.grad.abs().sum() > 0
    assert model.head.weight.grad is None
    if not quality and not uniqueness:
        torch.testing.assert_close(output["query_partial_q2s_logits"], output["query_partial_q2s_base_logits"])


def test_temporal_projection_called_once_spatial_inputs_are_conditioned():
    model = _new_model().eval()
    features, raw, meta = _episode()
    calls, spatial = [], []
    hook = model.blocks[0].temporal_attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    spatial_hook = model.blocks[0].attn.qkv.register_forward_pre_hook(lambda module, args: spatial.append(args[0].detach()))
    label_conditioned_episode(model, features, raw, meta)
    hook.remove()
    spatial_hook.remove()
    assert len(calls) == 1
    assert len(spatial) == 4  # Support + Query for each class, one video chunk each.
    assert not torch.allclose(spatial[0], spatial[2])


def test_raw_quality_is_unchanged_by_new_post_encoding():
    model = _new_model().eval()
    legacy = _model().eval()
    legacy.query_evidence_verifier.load_state_dict(model.query_evidence_verifier.state_dict())
    features, raw, meta = _episode()
    new = label_conditioned_episode(model, features, raw, meta)
    old = _run(legacy, features, raw, meta)
    for key in ("query_quality_logits", "query_quality_valid", "query_frame_absolute_patch_mass"):
        torch.testing.assert_close(new[key], old[key], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("class_chunk", [1, 2])
def test_query_labels_isolated_class_permutation_and_support_order(class_chunk):
    model = _new_model().eval()
    model.label_pre_temporal.class_chunk_size = class_chunk
    model.label_pre_temporal.batch_raw_quality = class_chunk > 1
    features, raw, meta = _episode()
    first = label_conditioned_episode(model, features, raw, meta)
    changed = copy.deepcopy(meta)
    changed["episode_positive_labels"] = changed["episode_positive_labels"].float()
    changed["episode_positive_labels"][2:] = float("nan")
    second = label_conditioned_episode(model, features, raw, changed)
    order = torch.tensor([1, 0])
    permuted = dict(meta, episode_class_ids=meta["episode_class_ids"][order],
                    episode_positive_labels=meta["episode_positive_labels"][:, order])
    third = label_conditioned_episode(model, features, raw, permuted)
    support_order = torch.tensor([1, 0, 2, 3])
    permuted = {key: (value[support_order] if value.ndim and value.shape[0] == 4 else value)
                for key, value in meta.items()}
    fourth = label_conditioned_episode(model, features[support_order], raw[support_order], permuted)
    for key in ("query_partial_q2s_logits", "query_quality_logits", "query_relation_logits"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        torch.testing.assert_close(first[key][:, order], third[key], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(first[key], fourth[key], atol=2e-6, rtol=2e-6)


def test_chunk_and_checkpoint_outputs_and_gradients_agree():
    a = _new_model(checkpoint=False)
    b = copy.deepcopy(a)
    b.label_pre_temporal.video_chunk_size = 1
    b.label_pre_temporal.activation_checkpoint = True
    b.query_evidence_verifier.activation_checkpoint = True
    features, raw, meta = _episode()
    x = features.clone().requires_grad_()
    y = features.clone().requires_grad_()
    first = label_conditioned_episode(a, x, raw, meta)
    second = label_conditioned_episode(b, y, raw, meta)
    for key in ("query_partial_q2s_logits", "query_quality_logits", "query_relation_logits"):
        torch.testing.assert_close(first[key], second[key], atol=2e-5, rtol=2e-5)
    _loss(a, first, meta).backward()
    _loss(b, second, meta).backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


@pytest.mark.parametrize("quality,uniqueness", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("checkpoint,inner_checkpoint,mlp_checkpoint,readout_checkpoint",
                         [(False, False, False, False), (True, True, False, False),
                          (True, False, False, False), (False, False, True, False), (False, False, True, True)])
@pytest.mark.parametrize("class_chunk", [2, 4])
@pytest.mark.parametrize("batch_quality", [False, True])
def test_batched_classes_match_serial_outputs_and_gradients(quality, uniqueness, checkpoint, inner_checkpoint,
                                                           mlp_checkpoint, readout_checkpoint, class_chunk, batch_quality):
    a = _new_model(quality, uniqueness, checkpoint)
    b = copy.deepcopy(a)
    b.label_pre_temporal.class_chunk_size = class_chunk
    b.label_pre_temporal.video_chunk_size = 1
    b.label_pre_temporal.batch_raw_quality = batch_quality
    b.blocks[0].checkpoint_mlp = mlp_checkpoint
    b.label_pre_temporal.readout_checkpoint = readout_checkpoint
    if b.query_evidence_verifier is not None:
        b.query_evidence_verifier.activation_checkpoint = inner_checkpoint
    features, raw, meta = _episode()
    # Three classes exercises a partial last group and C > episode class count.
    meta["episode_class_ids"] = torch.arange(3)
    meta["episode_positive_labels"] = torch.cat((meta["episode_positive_labels"],
                                               meta["episode_positive_labels"][:, 1:]), 1)
    x = features.clone().requires_grad_()
    y = features.clone().requires_grad_()
    calls = []
    hook = b.blocks[0].temporal_attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    first = label_conditioned_episode(a, x, raw, meta)
    second = label_conditioned_episode(b, y, raw, meta)
    for key, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[key], atol=2e-5, rtol=2e-5, msg=key)
    _loss(a, first, meta).backward()
    _loss(b, second, meta).backward()
    hook.remove()
    assert len(calls) == 1  # Temporal projections are not redone per class or checkpoint.
    torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5, msg=name)
    # Execution grouping does not change the checkpoint/state_dict contract.
    b.load_state_dict(a.state_dict(), strict=True)


def test_batched_raw_support_references_built_once_per_group(monkeypatch):
    model = _new_model()
    model.label_pre_temporal.class_chunk_size = 2
    model.label_pre_temporal.video_chunk_size = 1
    model.label_pre_temporal.batch_raw_quality = True
    quality = model.query_evidence_verifier.semantic_quality
    original, calls = quality._references, []
    def record(*args):
        calls.append(args[2].shape[0])
        return original(*args)
    monkeypatch.setattr(quality, "_references", record)
    features, raw, meta = _episode()
    label_conditioned_episode(model, features, raw, meta)
    assert calls == [2]  # Two candidate classes together, not per Query video.


@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("progress", [0., 0.4, 1.])
def test_batched_raw_quality_preserves_learned_memory_and_input_gradients(detach, progress):
    a = _new_model(checkpoint=True)
    a.query_evidence_verifier.detach_inputs = detach
    a.query_evidence_verifier.transport_progress = progress
    b = copy.deepcopy(a)
    b.label_pre_temporal.class_chunk_size = 2
    b.label_pre_temporal.batch_raw_quality = True
    b.query_evidence_verifier.activation_checkpoint = False
    features, raw, meta = _episode()
    first_raw, second_raw = raw.clone().requires_grad_(), raw.clone().requires_grad_()
    first = label_conditioned_episode(a, features, first_raw, meta)
    second = label_conditioned_episode(b, features, second_raw, meta)
    _loss(a, first, meta).backward()
    _loss(b, second, meta).backward()
    torch.testing.assert_close(first_raw.grad, second_raw.grad, atol=3e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5, msg=name)
    for module in (b.query_evidence_verifier.semantic_quality.projection,
                   b.query_evidence_verifier.semantic_quality.pool):
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("progress", [0., 0.4, 1.])
def test_diagnostic_output_boundary_preserves_all_training_gradients(monkeypatch, checkpoint, progress):
    import trokens.models.label_conditioned_routing as routing
    a = _new_model(checkpoint=checkpoint)
    a.query_evidence_verifier.transport_progress = progress
    b = copy.deepcopy(a)
    features, raw, meta = _episode()
    x = features.clone().requires_grad_()
    y = features.clone().requires_grad_()
    # Compare against the original output graph, not just the new contract.
    with monkeypatch.context() as original:
        original.setattr(routing, "_detach_diagnostics", lambda result: result)
        before = label_conditioned_episode(a, x, raw, meta)
    after = label_conditioned_episode(b, y, raw, meta)
    loss_keys = {"query_partial_q2s_logits", "query_partial_q2s_base_logits",
                 "query_quality_logits", "query_relation_logits"}
    for key, value in after.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            assert value.requires_grad == (key in loss_keys), key
    first, second = _loss(a, before, meta), _loss(b, after, meta)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    first.backward()
    second.backward()
    torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("empty", ["query", "support", "negative"])
@pytest.mark.parametrize("batch_quality", [False, True])
def test_invalid_observations_and_missing_references(empty, batch_quality):
    model = _new_model(checkpoint=True)
    model.label_pre_temporal.class_chunk_size = 2
    model.label_pre_temporal.batch_raw_quality = batch_quality
    features, raw, meta = _episode()
    if empty == "negative":
        meta["episode_label_known_mask"] = meta["episode_positive_labels"].clone()
    else:
        sl = slice(2, None) if empty == "query" else slice(None, 2)
        meta["pred_visibility"][sl] = False
        meta["pred_query_mask"][sl] = False
    result = label_conditioned_episode(model, features, raw, meta)
    _loss(model, result, meta).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if empty in ("negative", "support"):
        assert not result["query_relation_valid"].any()
        assert result["query_frame_matchability"].eq(1).all()


def test_explicit_output_targets_and_checkpoint_safety(tmp_path):
    model = _new_model()
    features, raw, meta = _episode()
    meta["sample_type"] = ["support", "support", "query", "query"]
    result = label_conditioned_episode(model, features, raw, meta)
    preds, patches, aux = unpack_episode_output(result)
    assert preds is None and patches is None and aux is result
    targets = single_path_episode_targets(result, meta["episode_positive_labels"], meta)
    torch.testing.assert_close(targets["query_batch_labels"], meta["episode_positive_labels"][2:].float())
    bad = dict(result, query_partial_query_sample_indices=torch.tensor([3, 2]))
    with pytest.raises(ValueError, match="order"):
        single_path_episode_targets(bad, meta["episode_positive_labels"], meta)
    state = model.state_dict()
    assert not validate_label_pre_temporal_checkpoint(model, state)
    legacy = {k: v for k, v in state.items() if not k.startswith("label_pre_temporal.")}
    with pytest.raises(ValueError, match="missing"):
        validate_label_pre_temporal_checkpoint(model, legacy)
    assert validate_label_pre_temporal_checkpoint(model, legacy, allow_init=True)
    partial_state = dict(state)
    del partial_state["blocks.0.temporal_attn.qkv.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_label_pre_temporal_checkpoint(model, partial_state, allow_init=True)
    path = tmp_path / "legacy.pyth"
    torch.save(dict(model_state=legacy, epoch=8, optimizer_state={}), path)
    assert load_checkpoint(str(path), model, data_parallel=False,
                           allow_label_pre_temporal_init=True) == -1
    path = tmp_path / "resume.pyth"
    torch.save(dict(model_state=state, epoch=8), path)
    assert load_checkpoint(str(path), model, data_parallel=False) == 8


def test_pointformer_forward_uses_raw_even_with_quality_off_and_skips_old_path():
    model = _new_model(quality=False)
    _, raw, meta = _episode()
    model.feat_extractor_type = "dinotxt_vitl14_reg4"
    model.cfg.MF.USE_BASE_POS_EMBED = True
    meta["pred_tracks"] = torch.zeros(4, 3, 5, 2)
    meta["hod_feat"] = torch.zeros_like(raw)
    model.get_dinotxt_features = lambda video: raw.unsqueeze(2)
    sampled = []
    def sample(feature, tracks, add_pt_pos_embed):
        sampled.append(feature)
        return feature.flatten(2, 3)
    model._sample_point_features = sample
    model.add_st_pos_embeddings = lambda feature: feature + 10
    model.hod_motion_module = lambda feature: torch.ones_like(raw)
    model.cross_motion_module = lambda tracks, visibility: torch.full_like(raw, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy global path must not run")
    model.pt_forward = forbidden
    model.head.forward = forbidden
    result = model(dict(video=torch.empty(4, 3, 3, 2, 2), metadata=meta))
    assert result["single_path_episode"] and len(sampled) == 2
    torch.testing.assert_close(sampled[0].flatten(2, 3), raw)
    torch.testing.assert_close(sampled[1].flatten(2, 3), raw + 10)


class _EpisodeHarness(nn.Module):
    def __init__(self, classification=False, post_readout=False, scoring="projected_cosine"):
        super().__init__()
        self.core = _new_model(checkpoint=True, classification=classification,
                               post_readout=post_readout, scoring=scoring)

    def forward(self, inputs):
        features = inputs["video"]
        return label_conditioned_episode(self.core, features, features.detach(), inputs["metadata"])


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("scoring", ["projected_cosine", "cosine"])
def test_train_validation_and_test_entrypoints_without_legacy_outputs(tmp_path, monkeypatch, classification, scoring):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from tools import train_few_shot as train, test_few_shot as evaluate
    from trokens.utils.meters import TrainMeter, ValMeter
    model = _EpisodeHarness(classification=classification, scoring=scoring,
                             post_readout=scoring == "cosine")
    cfg = model.core.cfg
    cfg.NUM_GPUS = 0
    cfg.TRAIN.MIXED_PRECISION = False
    cfg.MODEL.NUM_CLASSES = 2
    cfg.TEST.SEEN_LABELS = [0]
    cfg.TEST.NOVEL_LABELS = [1]
    cfg.OUTPUT_DIR = str(tmp_path)
    cfg["wandb"] = None
    cfg["csv_dump_name"] = "single_path.csv"
    features, _, meta = _episode()
    meta["sample_type"] = ["support", "support", "query", "query"]
    meta["video_name"] = ["s0", "s1", "q0", "q1"]
    labels = meta["episode_positive_labels"].float()
    batch = [(features, labels, torch.arange(4), meta)]
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy classification/split must be bypassed")
    for module in (train, evaluate):
        monkeypatch.setattr(module, "create_progress_bar", lambda *args: None)
        monkeypatch.setattr(module, "support_query_split_multilabel", forbidden)
    if not classification:
        monkeypatch.setattr(train, "multilabel_classification_loss", forbidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    watched = (model.core.label_pre_temporal.query.weight if scoring == "projected_cosine"
               else model.core.blocks[0].temporal_attn.qkv.weight)
    before = watched.detach().clone()
    head_before = model.core.head.weight.detach().clone()
    train.train_epoch(batch, model, optimizer, train.make_grad_scaler(False),
                      TrainMeter(1, cfg), 0, cfg)
    assert not torch.equal(before, watched)
    assert (not torch.equal(head_before, model.core.head.weight)) == classification
    score = train.eval_epoch(batch, model, ValMeter(1, cfg), 0, cfg)
    assert 0 <= score <= 100
    evaluate.test_epoch(batch, model, ValMeter(1, cfg), 0, cfg)
    assert (tmp_path / "single_path.csv").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint,class_chunk,mlp_checkpoint,readout_checkpoint",
                         [(True, 1, False, False), (True, 2, False, False), (False, 2, False, False),
                          (False, 2, True, False), (False, 2, True, True)])
def test_cuda_amp_checkpoint_forward_backward_and_optimizer(dtype, checkpoint, class_chunk, mlp_checkpoint, readout_checkpoint):
    model = _new_model(checkpoint=checkpoint).cuda()
    model.label_pre_temporal.class_chunk_size = class_chunk
    model.blocks[0].checkpoint_mlp = mlp_checkpoint
    model.label_pre_temporal.readout_checkpoint = readout_checkpoint
    if class_chunk > 1:
        model.label_pre_temporal.batch_raw_quality = True
        model.query_evidence_verifier.activation_checkpoint = False
    features, raw, meta = _episode("cuda")
    features.requires_grad_()
    meta["pred_query_mask"][2, 1] = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):
        result = label_conditioned_episode(model, features, raw, meta)
        loss = _loss(model, result, meta)
    before = model.label_pre_temporal.query.weight.detach().clone()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.label_pre_temporal.query.weight)


def _single_path_ddp_worker(rank, rendezvous, cuda=False, class_chunk=1, checkpoint=True,
                            mlp_checkpoint=False, readout_checkpoint=False, classification=False,
                            post_readout=False, scoring="projected_cosine"):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(_EpisodeHarness(classification=classification,
                                                        post_readout=post_readout, scoring=scoring).to(device),
                                        device_ids=[rank] if cuda else None,
                                        find_unused_parameters=True)
        model.module.core.label_pre_temporal.class_chunk_size = class_chunk
        model.module.core.label_pre_temporal.batch_raw_quality = class_chunk > 1
        model.module.core.label_pre_temporal.activation_checkpoint = checkpoint
        model.module.core.blocks[0].checkpoint_mlp = mlp_checkpoint
        model.module.core.label_pre_temporal.readout_checkpoint = readout_checkpoint
        model.module.core.query_evidence_verifier.activation_checkpoint = checkpoint and class_chunk == 1
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=cuda)
        head_before = model.module.core.head.weight.detach().clone()
        for step in range(2):
            model.module.core.query_evidence_verifier.transport_progress = step * .002
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                features, _, meta = _episode(device)
                features = features + rank * .1 + micro * .01
                if rank == 1 and step == 0:
                    meta["pred_query_mask"][2:] = False
                with model.no_sync() if micro == 0 else nullcontext():
                    with torch.autocast("cuda", dtype=torch.float16) if cuda else nullcontext():
                        result = model(dict(video=features, metadata=meta))
                        loss = _loss(model.module.core, result, meta) / 2
                    scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            scaler.step(optimizer)
            scaler.update()
        if classification:
            assert not torch.equal(head_before, model.module.core.head.weight)
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        replicas = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(replicas, flat)
        torch.testing.assert_close(replicas[0], replicas[1], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("checkpoint,class_chunk,mlp_checkpoint,readout_checkpoint",
                         [(True, 1, False, False), (True, 2, False, False), (False, 2, False, False),
                          (False, 2, True, False), (False, 2, True, True)])
def test_two_rank_single_path_ddp_checkpoint_and_accumulation(tmp_path, checkpoint, class_chunk, mlp_checkpoint, readout_checkpoint):
    torch.multiprocessing.spawn(_single_path_ddp_worker,
                                args=((tmp_path / "rendezvous").as_uri(), False, class_chunk, checkpoint,
                                      mlp_checkpoint, readout_checkpoint),
                                nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available(),
                    reason="Two visible CUDA GPUs and NCCL required")
@pytest.mark.parametrize("checkpoint,class_chunk,mlp_checkpoint,readout_checkpoint",
                         [(True, 1, False, False), (True, 2, False, False), (False, 2, False, False),
                          (False, 2, True, False), (False, 2, True, True)])
def test_two_gpu_amp_ddp_checkpoint_accumulation_and_zero_warmup(tmp_path, checkpoint, class_chunk, mlp_checkpoint, readout_checkpoint):
    torch.multiprocessing.spawn(_single_path_ddp_worker,
                                args=((tmp_path / "cuda_rendezvous").as_uri(), True, class_chunk, checkpoint,
                                      mlp_checkpoint, readout_checkpoint),
                                nprocs=2, join=True)
