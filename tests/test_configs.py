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


def test_sav_explicitly_defines_active_early_profile_and_cosine_alias_matches():
    with (CONFIGS / "sav.yaml").open() as stream:
        explicit = yaml.safe_load(stream)
    assert "_BASE_" not in explicit
    expected = dict(
        ENABLE=True, SCORING="cosine", ROUTE_TEMPERATURE=.04, GATE_STRENGTH=.25,
        TEMPORAL_BIAS_SCALE=1., SPATIAL_BIAS_SCALE=1., CLASS_CHUNK_SIZE=5,
        VIDEO_CHUNK_SIZE=8, ACTIVATION_CHECKPOINT=False, BATCH_RAW_QUALITY=True,
        MLP_CHECKPOINT=True, READOUT_CHECKPOINT=True,
    )
    assert explicit["FEW_SHOT"]["LABEL_PRE_TEMPORAL"] == expected
    active, alias = get_cfg(), get_cfg()
    active.merge_from_file(str(CONFIGS / "sav.yaml"))
    alias.merge_from_file(str(CONFIGS / "sav_label_pre_temporal_cosine.yaml"))
    assert active == alias
    assert active.FEW_SHOT.CLASS_LOSS_LAMBDA == 1.
    assert not active.TRAIN.AUTO_RESUME and not active.TRAIN.CHECKPOINT_FILE_PATH


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
