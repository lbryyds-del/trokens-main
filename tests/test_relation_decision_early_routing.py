"""Direct quality decisions composed with the unchanged early routing path."""

import copy
from pathlib import Path

import pytest
import torch

from test_learned_query_verifier import _episode, _model, _run
from test_label_pre_temporal import _new_model, _loss, _single_path_ddp_worker
from trokens.config.defaults import get_cfg
from trokens.models.label_conditioned_routing import label_conditioned_episode
from trokens.utils.checkpoint import load_checkpoint


def _decision(**kwargs):
    return _new_model(quality_mode="relation_decision", scoring="cosine",
                      classification=True, **kwargs)


def test_early_configs_inherit_new_quality_without_changing_routing_contract():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/trokens").glob("sav_label_pre_temporal*.yaml"))
    assert len(paths) == 5
    paths.append(root / "configs/trokens/sav.yaml")
    for path in paths:
        cfg = get_cfg()
        cfg.merge_from_file(str(path))
        early, match = cfg.FEW_SHOT.LABEL_PRE_TEMPORAL, cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
        assert early.ENABLE and early.TEMPORAL_BIAS_SCALE == early.SPATIAL_BIAS_SCALE == 1.
        assert match.LEARNED_VERIFIER.QUALITY_MODE == "relation_decision"
        assert match.QUALITY_ENABLE and match.UNIQUENESS_ENABLE
        assert cfg.FEW_SHOT.CLASS_LOSS_LAMBDA == 1.
        assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == (.6, .4)
        assert "relation_decision" in cfg.OUTPUT_DIR
        assert "post_rho" in cfg.OUTPUT_DIR
        assert "reuse_pre_weights" in cfg.OUTPUT_DIR
        assert "POST_SUPPORT_READOUT" not in early
        assert not cfg.TRAIN.AUTO_RESUME and not cfg.TRAIN.CHECKPOINT_FILE_PATH
        if path.stem == "sav" or path.stem.endswith("cosine"):
            assert early.SCORING == "cosine"
            assert (early.CLASS_CHUNK_SIZE, early.VIDEO_CHUNK_SIZE) == (5, 8)
            assert early.BATCH_RAW_QUALITY and early.MLP_CHECKPOINT and early.READOUT_CHECKPOINT
            assert not early.ACTIVATION_CHECKPOINT and not match.LEARNED_VERIFIER.ACTIVATION_CHECKPOINT


@pytest.mark.parametrize("quality,uniqueness", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("batch_quality", [False, True])
def test_only_quality_changes_not_early_encoding_prototypes_or_rho(quality, uniqueness, batch_quality):
    old = _new_model(quality=quality, uniqueness=uniqueness, scoring="cosine",
                     classification=True)
    new = _decision(quality=quality, uniqueness=uniqueness)
    # Compare the same encoder, not differences caused by RNG consumption while
    # constructing differently sized heads in this miniature test harness.
    shared = {k: v for k, v in old.state_dict().items()
              if not k.startswith("query_evidence_verifier.semantic_quality.")
              and k != "query_evidence_verifier.architecture_version"}
    new.load_state_dict(shared, strict=False)
    for model in (old, new):
        model.label_pre_temporal.batch_raw_quality = batch_quality
        model.label_pre_temporal.class_chunk_size = 2
    features, raw, meta = _episode()
    features.requires_grad_()
    before = label_conditioned_episode(old, features, raw, meta)
    after = label_conditioned_episode(new, features, raw, meta)
    for key in ("query_partial_query_prototypes", "query_partial_support_prototypes",
                "query_partial_q2s_base_logits", "query_patch_conditional_weights",
                "query_relation_logits", "global_classification_logits"):
        if key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0, msg=key)
    if quality:
        assert not torch.equal(before["query_quality_logits"], after["query_quality_logits"])
        assert "query_quality_relation_feature_norm" in after
        assert "query_quality_correction" not in after
    _loss(new, after, meta).backward()
    for name, p in new.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("batch_quality", [False, True])
def test_raw_decision_matches_late_path_and_query_labels_remain_isolated(batch_quality):
    new = _decision().eval()
    new.label_pre_temporal.batch_raw_quality = batch_quality
    new.label_pre_temporal.class_chunk_size = 2
    late = _model(quality_mode="relation_decision").eval()
    late.query_evidence_verifier.load_state_dict(new.query_evidence_verifier.state_dict(), strict=True)
    features, raw, meta = _episode()
    early = label_conditioned_episode(new, features, raw, meta)
    previous = _run(late, features, raw, meta)
    for key in ("query_quality_logits", "query_quality_valid", "query_frame_absolute_patch_mass"):
        torch.testing.assert_close(early[key], previous[key], atol=2e-6, rtol=2e-6)
    changed = copy.deepcopy(meta)
    changed["episode_positive_labels"] = changed["episode_positive_labels"].float()
    changed["episode_positive_labels"][2:] = float("nan")
    isolated = label_conditioned_episode(new, features, raw, changed)
    for key in ("query_quality_logits", "query_relation_logits", "query_partial_q2s_logits"):
        torch.testing.assert_close(early[key], isolated[key], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["batch", "outer_checkpoint", "selective_checkpoint"])
def test_new_decision_serial_batched_checkpoint_outputs_and_gradients(mode):
    a = _decision()
    b = copy.deepcopy(a)
    b.label_pre_temporal.batch_raw_quality = True
    b.label_pre_temporal.class_chunk_size = 2
    b.label_pre_temporal.video_chunk_size = 1
    if mode == "outer_checkpoint":
        b.label_pre_temporal.activation_checkpoint = True
        b.query_evidence_verifier.activation_checkpoint = True
    elif mode == "selective_checkpoint":
        b.blocks[0].checkpoint_mlp = True
        b.label_pre_temporal.readout_checkpoint = True
    features, raw, meta = _episode()
    x, y = features.clone().requires_grad_(), features.clone().requires_grad_()
    first = label_conditioned_episode(a, x, raw, meta)
    second = label_conditioned_episode(b, y, raw, meta)
    for key, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[key], atol=3e-5, rtol=3e-5, msg=key)
    _loss(a, first, meta).backward()
    _loss(b, second, meta).backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5, msg=name)


def test_checkpoint_roundtrip_and_cross_quality_or_cross_routing_rejection(tmp_path):
    model = _decision()
    state = model.state_dict()
    path = tmp_path / "early_decision.pyth"
    torch.save(dict(model_state=state, epoch=3), path)
    assert load_checkpoint(str(path), _decision(), data_parallel=False) == 3
    # Early-routing validation checks all trainable keys before the verifier
    # version guard, so it may reject the changed head at that earlier check.
    with pytest.raises(ValueError, match="architecture|Incomplete Label-pre-temporal"):
        load_checkpoint(str(path), _new_model(scoring="cosine", classification=True),
                        data_parallel=False)
    with pytest.raises(ValueError, match="LABEL_PRE_TEMPORAL"):
        load_checkpoint(str(path), _model(quality_mode="relation_decision"), data_parallel=False)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two idle CUDA GPUs required")
@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_two_gpu_amp_ddp_new_quality_with_early_cosine(tmp_path, outer_checkpoint):
    torch.multiprocessing.spawn(
        _single_path_ddp_worker,
        args=("file://" + str(tmp_path / "ddp_init"), True, 2, outer_checkpoint,
              not outer_checkpoint, not outer_checkpoint, True, "cosine", "relation_decision"),
        nprocs=2, join=True)
