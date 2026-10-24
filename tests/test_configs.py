"""Maintained entrypoints and episodic batching after config cleanup."""

from pathlib import Path

import pytest
import torch
import yaml

from trokens.config.defaults import assert_and_infer_cfg, get_cfg


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "trokens"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_dataset_config_merges(path):
    cfg = get_cfg()
    cfg.merge_from_file(str(path))
    assert_and_infer_cfg(cfg)


def test_sav_defines_concat_profile_and_preserves_historical_cosine_control():
    assert get_cfg().FEW_SHOT.TEXT_CONCAT_ST.SPATIAL_TEXT_TOKENS == 1
    with (CONFIGS / "sav.yaml").open() as stream:
        explicit = yaml.safe_load(stream)
    assert "_BASE_" not in explicit
    expected = dict(
        ENABLE=True, TEMPORAL_TEXT_TOKENS=1, SPATIAL_TEXT_TOKENS=1,
        READOUT='spatial_text_attention', CLASS_CHUNK_SIZE=5,
        VIDEO_CHUNK_SIZE=8, ACTIVATION_CHECKPOINT=False,
        MLP_CHECKPOINT=True, READOUT_CHECKPOINT=True, CACHE_FROZEN_TEXT=True,
    )
    assert explicit["FEW_SHOT"]["TEXT_CONCAT_ST"] == expected
    active, alias = get_cfg(), get_cfg()
    active.merge_from_file(str(CONFIGS / "sav.yaml"))
    alias.merge_from_file(str(CONFIGS / "sav_label_pre_temporal_cosine.yaml"))
    assert not active.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE
    assert not active.FEW_SHOT.POT_ROUTE.ENABLE
    assert active.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE
    assert not active.FEW_SHOT.SUPPORT_TEXT_FUSION.ENABLE
    assert active.FEW_SHOT.QUERY_CLASS_MATCHABILITY.QUALITY_ENABLE
    assert active.FEW_SHOT.QUERY_CLASS_MATCHABILITY.UNIQUENESS_ENABLE
    assert alias.FEW_SHOT.LABEL_PRE_TEMPORAL.ENABLE
    assert alias.FEW_SHOT.LABEL_PRE_TEMPORAL.SCORING == 'cosine'
    assert not alias.FEW_SHOT.TEXT_CONCAT_ST.ENABLE
    assert alias.FEW_SHOT.POT_ROUTE.ENABLE and alias.FEW_SHOT.SUPPORT_TEXT_FUSION.ENABLE
    assert alias.FEW_SHOT.QUERY_CLASS_MATCHABILITY.QUALITY_ENABLE
    assert alias.FEW_SHOT.QUERY_CLASS_MATCHABILITY.UNIQUENESS_ENABLE
    for name in ('DATA', 'MODEL', 'MF', 'POINT_INFO', 'SOLVER'):
        assert active[name] == alias[name]
    assert active.FEW_SHOT.CLASS_LOSS_LAMBDA == 1.
    assert not active.TRAIN.AUTO_RESUME and not active.TRAIN.CHECKPOINT_FILE_PATH


def test_s32_control_preserves_historical_execution_and_loss_settings():
    active, control = get_cfg(), get_cfg()
    active.merge_from_file(str(CONFIGS / 'sav.yaml'))
    control.merge_from_file(str(CONFIGS / 'sav_text_concat_s32.yaml'))
    assert control.FEW_SHOT.TEXT_CONCAT_ST.SPATIAL_TEXT_TOKENS == 32
    assert not control.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT
    assert control.FEW_SHOT.TEXT_CONCAT_ST.CLASS_CHUNK_SIZE == 2
    assert control.FEW_SHOT.TEXT_CONCAT_ST.VIDEO_CHUNK_SIZE == 4
    assert control.FEW_SHOT.TEXT_CONCAT_ST.ACTIVATION_CHECKPOINT
    assert not control.FEW_SHOT.TEXT_CONCAT_ST.MLP_CHECKPOINT
    assert not control.FEW_SHOT.TEXT_CONCAT_ST.READOUT_CHECKPOINT
    assert active.OUTPUT_DIR != control.OUTPUT_DIR
    control.FEW_SHOT.TEXT_CONCAT_ST.SPATIAL_TEXT_TOKENS = 1
    control.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = True
    for name in ('CLASS_CHUNK_SIZE', 'VIDEO_CHUNK_SIZE', 'ACTIVATION_CHECKPOINT',
                 'MLP_CHECKPOINT', 'READOUT_CHECKPOINT'):
        control.FEW_SHOT.TEXT_CONCAT_ST[name] = active.FEW_SHOT.TEXT_CONCAT_ST[name]
    match = control.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert not match.BASE_ONLY_LOSS_ENABLE and match.DUAL_LOGIT_LOSS_ENABLE
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == (.6, .4)
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == .05
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == .05
    control.FEW_SHOT.QUERY_CLASS_MATCHABILITY = active.FEW_SHOT.QUERY_CLASS_MATCHABILITY.clone()
    control.OUTPUT_DIR = active.OUTPUT_DIR
    assert control == active


def test_single_text_fast_profile_changes_only_execution_and_output():
    active, fast = get_cfg(), get_cfg()
    active.merge_from_file(str(CONFIGS / 'sav_text_concat_s1_base.yaml'))
    fast.merge_from_file(str(CONFIGS / 'sav_text_concat_s1_fast.yaml'))
    assert fast.FEW_SHOT.TEXT_CONCAT_ST.CLASS_CHUNK_SIZE == 5
    assert fast.FEW_SHOT.TEXT_CONCAT_ST.VIDEO_CHUNK_SIZE == 8
    assert not fast.FEW_SHOT.TEXT_CONCAT_ST.ACTIVATION_CHECKPOINT
    assert fast.FEW_SHOT.TEXT_CONCAT_ST.MLP_CHECKPOINT
    assert fast.FEW_SHOT.TEXT_CONCAT_ST.READOUT_CHECKPOINT
    assert fast.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT
    assert fast.OUTPUT_DIR != active.OUTPUT_DIR
    for name in ('CLASS_CHUNK_SIZE', 'VIDEO_CHUNK_SIZE', 'ACTIVATION_CHECKPOINT',
                 'MLP_CHECKPOINT', 'READOUT_CHECKPOINT'):
        fast.FEW_SHOT.TEXT_CONCAT_ST[name] = active.FEW_SHOT.TEXT_CONCAT_ST[name]
    fast.OUTPUT_DIR = active.OUTPUT_DIR
    assert fast == active


def test_base_control_has_exactly_global_plus_base_objective():
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIGS / 'sav_text_concat_s1_base.yaml'))
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert cfg.FEW_SHOT.CLASS_LOSS_LAMBDA == cfg.FEW_SHOT.Q2S_LOSS_LAMBDA == 1.
    assert match.ENABLE  # Keep unknown-label masking, not the verifier heads.
    assert match.BASE_ONLY_LOSS_ENABLE and not match.DUAL_LOGIT_LOSS_ENABLE
    assert not match.QUALITY_ENABLE and not match.UNIQUENESS_ENABLE
    assert match.BASE_LOGIT_LOSS_WEIGHT == 1. and match.VERIFIED_LOGIT_LOSS_WEIGHT == 0.
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == 0.
    assert match.LEARNED_VERIFIER.QUALITY_RELATION_STRENGTH == 0.
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == 0.
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == 0.


def test_active_concat_verifier_changes_only_judgments_losses_and_output():
    active, base = get_cfg(), get_cfg()
    active.merge_from_file(str(CONFIGS / 'sav.yaml'))
    base.merge_from_file(str(CONFIGS / 'sav_text_concat_s1_base.yaml'))
    match = active.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert match.QUALITY_ENABLE and match.UNIQUENESS_ENABLE
    assert not match.BASE_ONLY_LOSS_ENABLE and match.DUAL_LOGIT_LOSS_ENABLE
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == (.6, .4)
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == .05
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == .05
    assert match.LEARNED_VERIFIER.DETACH_INPUTS
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == 1.
    assert match.ABSOLUTE_MASS_ONE_SIDED
    assert match.LEARNED_VERIFIER.QUALITY_MODE == 'relation_decision'
    assert active.OUTPUT_DIR != base.OUTPUT_DIR
    active.FEW_SHOT.QUERY_CLASS_MATCHABILITY = base.FEW_SHOT.QUERY_CLASS_MATCHABILITY.clone()
    active.OUTPUT_DIR = base.OUTPUT_DIR
    assert active == base


def test_few_shot_loader_uses_episode_sampler_instead_of_generic_batch_size(monkeypatch):
    from trokens.datasets import loader

    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIGS / "sav.yaml"))
    cfg.NUM_GPUS = 3
    cfg.TRAIN.BATCH_SIZE = cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 0
    assert_and_infer_cfg(cfg)
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    monkeypatch.setattr(loader, "build_dataset", lambda *args: dataset)
    monkeypatch.setattr(loader, "FewShotEpisodeSampler", lambda *args: [[0, 1, 2, 3]])
    monkeypatch.setattr(loader.utils, "loader_worker_init_fn", lambda dataset: None)

    for split in ("train", "val", "test"):
        batches = list(loader.construct_loader(cfg, split))
        assert len(batches) == 1
        torch.testing.assert_close(batches[0][0], torch.arange(4))
