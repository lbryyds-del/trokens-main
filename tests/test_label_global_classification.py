"""Restored global supervision without an additional video encoder."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from test_label_pre_temporal import _new_model, _loss, _single_path_ddp_worker
from test_learned_query_verifier import _episode
from trokens.models.label_conditioned_routing import label_conditioned_episode
from trokens.utils.checkpoint import load_checkpoint, validate_label_pre_temporal_checkpoint
from tools.few_shot_multilabel import multilabel_classification_loss, unpack_episode_output
from tools.few_shot_multilabel import compute_query_partial_q2s_loss
from trokens.config.defaults import get_cfg


@pytest.mark.parametrize("config_name", ["sav_label_pre_temporal.yaml",
                                       "sav_label_pre_temporal_parallel.yaml",
                                       "sav_label_pre_temporal_fast.yaml",
                                       "sav_label_pre_temporal_post_readout.yaml",
                                       "sav_label_pre_temporal_cosine.yaml"])
def test_total_objective_and_gradients_equal_legacy_five_term_loss(config_name):
    """Independent formula oracle: same predictions/targets => same old loss."""
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs/trokens" / config_name))
    cfg.TEST.SEEN_LABELS = [0, 2]
    cfg.SOLVER.TEMPRATURE = .73
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert cfg.FEW_SHOT.CLASS_LOSS_LAMBDA == cfg.FEW_SHOT.Q2S_LOSS_LAMBDA == 1.
    assert match.BASE_LOGIT_LOSS_WEIGHT == .6 and match.VERIFIED_LOGIT_LOSS_WEIGHT == .4
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == .05
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == .05
    torch.manual_seed(93)
    global_logits, base, verified, quality, relation = inputs = tuple(
        torch.randn(*shape, requires_grad=True)
        for shape in ((4, 3), (2, 2), (2, 2), (2, 2, 3), (2, 2, 3)))
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 1., 1.]])
    targets = torch.tensor([[1., 0.], [-1., 1.]])
    valid = torch.tensor([[[True, False, True], [False, False, False]],
                          [[True, True, True], [False, True, True]]])
    aux = dict(query_partial_q2s_base_logits=base, query_quality_logits=quality,
               query_relation_logits=relation, query_quality_valid=valid, query_relation_valid=valid)
    q2s, _, _ = compute_query_partial_q2s_loss(verified, targets, aux, cfg)
    cls = multilabel_classification_loss(global_logits / cfg.SOLVER.TEMPRATURE,
                                         labels, cfg, nn.BCEWithLogitsLoss())
    actual = cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * cls + cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s

    bce = nn.BCEWithLogitsLoss()
    known = targets >= 0
    def legacy_mil(logits):
        bag_logits, bag_labels = [], []
        for q in range(2):
            for k in range(2):
                if known[q, k] and valid[q, k].any():
                    frames = logits[q, k][valid[q, k]]
                    bag_logits.append(torch.logsumexp(frames, 0) - frames.new_tensor(frames.numel()).log())
                    bag_labels.append(targets[q, k])
        return bce(torch.stack(bag_logits), torch.stack(bag_labels))
    expected = (bce(global_logits[:, [0, 2]] / .73, labels[:, [0, 2]])
                + .6 * bce(base[known] / .73, targets[known])
                + .4 * bce(verified[known] / .73, targets[known])
                + .05 * legacy_mil(quality) + .05 * legacy_mil(relation))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    for a, b in zip(torch.autograd.grad(actual, inputs, retain_graph=True),
                    torch.autograd.grad(expected, inputs)):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("cls,patch_cls", [(True, False), (True, True), (False, False)])
def test_neutral_global_readout_matches_original_classifier(cls, patch_cls):
    model = _new_model(cls=cls, classification=True).eval()
    model.cfg.MODEL.USE_PATCH_AS_CLS = patch_cls
    model.label_pre_temporal.strength = 0.
    features, raw, meta = _episode()
    expected = model.head(model.head_drop(model.pt_forward(features, meta)[0]))
    calls = []
    hook = model.blocks[0].temporal_attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    result = label_conditioned_episode(model, features, raw, meta)
    hook.remove()
    torch.testing.assert_close(result["global_classification_logits"], expected, atol=2e-6, rtol=2e-6)
    assert len(calls) == 1
    preds, patches, aux = unpack_episode_output(result)
    assert preds is result["global_classification_logits"] and patches is None and aux is result


def test_restoring_classifier_does_not_change_q2s_outputs_or_add_encoder_calls():
    model = _new_model(classification=True).eval()
    disabled = copy.deepcopy(model)
    disabled.label_pre_temporal.global_classifier = False
    features, raw, meta = _episode()
    calls = []
    hook = model.blocks[0].attn.qkv.register_forward_hook(lambda *args: calls.append(1))
    first = label_conditioned_episode(model, features, raw, meta)
    hook.remove()
    second = label_conditioned_episode(disabled, features, raw, meta)
    assert len(calls) == 4  # Two classes x (Support + Query); no neutral pass.
    for name in ("query_partial_q2s_base_logits", "query_partial_q2s_logits",
                 "query_quality_logits", "query_relation_logits"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert first["global_classification_logits"].requires_grad
    assert "global_classification_logits" not in second


def _three_class_episode():
    features, raw, meta = _episode()
    meta["episode_class_ids"] = torch.arange(3)
    meta["episode_positive_labels"] = torch.tensor(
        [[1., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 1., 1.]])
    return features, raw, meta


def _three_class_model():
    model = _new_model(classification=True)
    model.head = nn.Linear(4, 3)
    model.cfg.TEST.SEEN_LABELS = [0, 1, 2]
    return model


@pytest.mark.parametrize("checkpoint,class_chunk,selective", [
    (True, 2, False), (False, 2, True), (False, 4, False)])
def test_global_and_total_loss_chunk_checkpoint_output_gradient_equivalence(checkpoint, class_chunk, selective):
    a = _three_class_model()
    b = copy.deepcopy(a)
    b.label_pre_temporal.video_chunk_size = 1
    b.label_pre_temporal.class_chunk_size = class_chunk
    b.label_pre_temporal.batch_raw_quality = True
    b.label_pre_temporal.activation_checkpoint = checkpoint
    b.blocks[0].checkpoint_mlp = selective
    b.label_pre_temporal.readout_checkpoint = selective
    features, raw, meta = _three_class_episode()
    x, y = features.clone().requires_grad_(), features.clone().requires_grad_()
    first = label_conditioned_episode(a, x, raw, meta)
    second = label_conditioned_episode(b, y, raw, meta)
    torch.testing.assert_close(first["global_classification_logits"], second["global_classification_logits"],
                               atol=2e-6, rtol=2e-6)
    la, lb = _loss(a, first, meta), _loss(b, second, meta)
    torch.testing.assert_close(la, lb, atol=2e-6, rtol=2e-6)
    la.backward()
    lb.backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


def test_global_output_is_candidate_order_invariant_and_query_truth_isolated():
    model = _three_class_model().eval()
    model.label_pre_temporal.class_chunk_size = 2
    features, raw, meta = _three_class_episode()
    first = label_conditioned_episode(model, features, raw, meta)
    changed = copy.deepcopy(meta)
    changed["episode_positive_labels"][2:] = float("nan")
    second = label_conditioned_episode(model, features, raw, changed)
    order = torch.tensor([2, 0, 1])
    changed = dict(meta, episode_class_ids=meta["episode_class_ids"][order],
                   episode_positive_labels=meta["episode_positive_labels"][:, order])
    third = label_conditioned_episode(model, features, raw, changed)
    videos = torch.tensor([1, 0, 3, 2])
    changed = {key: (value[videos] if value.ndim and value.shape[0] == 4 else value)
               for key, value in meta.items()}
    fourth = label_conditioned_episode(model, features[videos], raw[videos], changed)
    key = "global_classification_logits"
    torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    torch.testing.assert_close(first[key], third[key], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(first[key][videos], fourth[key], atol=2e-6, rtol=2e-6)


def test_global_bce_supervises_all_videos_only_seen_classes_and_trains_router():
    model = _new_model(classification=True)
    model.cfg.TEST.SEEN_LABELS = [0]
    features, raw, meta = _episode()
    features.requires_grad_()
    output = label_conditioned_episode(model, features, raw, meta)
    logits = output["global_classification_logits"]
    logits.retain_grad()
    loss = multilabel_classification_loss(logits, meta["episode_positive_labels"],
                                          model.cfg, nn.BCEWithLogitsLoss())
    loss.backward()
    assert (logits.grad[:, 0].abs() > 0).all()  # Both Support and Query.
    assert logits.grad[:, 1].eq(0).all()  # No negative-only Novel supervision.
    for name in ("head.weight", "pre_logits.weight", "label_pre_temporal.query.weight",
                 "label_pre_temporal.key.weight", "blocks.0.temporal_attn.qkv.weight",
                 "blocks.0.attn.qkv.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert features.grad[:2].abs().sum() > 0 and features.grad[2:].abs().sum() > 0


@pytest.mark.parametrize("source_version", [1, 2, 3, 4, 5])
def test_classifier_checkpoint_requires_explicit_initialization_and_v6_resumes(tmp_path, source_version):
    old = _new_model()
    model = _new_model(classification=True)
    old_state = old.state_dict()
    old_state["label_pre_temporal.architecture_version"] = torch.tensor(source_version)
    with pytest.raises(ValueError, match="CHECKPOINT_EPOCH_RESET"):
        validate_label_pre_temporal_checkpoint(model, old_state)
    incomplete = dict(old_state)
    del incomplete["head.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_label_pre_temporal_checkpoint(model, incomplete, allow_init=True)
    old_path = tmp_path / "no_classifier.pyth"
    torch.save(dict(model_state=old_state, epoch=8, optimizer_state={}, scaler_state={}), old_path)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    assert load_checkpoint(str(old_path), model, data_parallel=False, optimizer=optimizer,
                           allow_label_pre_temporal_init=True) == -1
    assert model.label_pre_temporal.architecture_version.item() == 6
    assert model.head.weight.requires_grad and not optimizer.state
    features, raw, meta = _episode()
    output = label_conditioned_episode(model, features, raw, meta)
    _loss(model, output, meta).backward()
    optimizer.step()
    state = model.state_dict()
    assert not validate_label_pre_temporal_checkpoint(model, state)
    path = tmp_path / "with_classifier.pyth"
    torch.save(dict(model_state=state, epoch=2, optimizer_state=optimizer.state_dict()), path)
    restored = _new_model(classification=True)
    opt = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-3)
    assert load_checkpoint(str(path), restored, data_parallel=False, optimizer=opt) == 2
    assert opt.state and restored.label_pre_temporal.architecture_version.item() == 6
    torch.testing.assert_close(restored.head.weight, model.head.weight, atol=0, rtol=0)
    assert opt.state[restored.head.weight]["step"] == optimizer.state[model.head.weight]["step"]


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_global_classifier_two_rank_ddp_and_accumulation(tmp_path, checkpoint, selective):
    torch.multiprocessing.spawn(
        _single_path_ddp_worker,
        args=((tmp_path / "global_rendezvous").as_uri(), False, 2, checkpoint, selective, selective, True),
        nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available(),
                    reason="Two CUDA GPUs and NCCL required")
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_global_classifier_two_gpu_amp_ddp_and_accumulation(tmp_path, checkpoint, selective):
    torch.multiprocessing.spawn(
        _single_path_ddp_worker,
        args=((tmp_path / "global_cuda_rendezvous").as_uri(), True, 2, checkpoint, selective, selective, True),
        nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint,selective", [(True, False), (False, True)])
def test_global_classifier_cuda_amp_gradient_and_update(dtype, checkpoint, selective):
    model = _new_model(classification=True, checkpoint=checkpoint).cuda()
    model.label_pre_temporal.class_chunk_size = 2
    model.label_pre_temporal.batch_raw_quality = True
    model.blocks[0].checkpoint_mlp = selective
    model.label_pre_temporal.readout_checkpoint = selective
    features, raw, meta = _episode("cuda")
    meta["pred_query_mask"][2, 1] = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=dtype == torch.float16)
    before = model.head.weight.detach().clone()
    with torch.autocast("cuda", dtype=dtype):
        output = label_conditioned_episode(model, features, raw, meta)
        loss = _loss(model, output, meta)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.head.weight)
