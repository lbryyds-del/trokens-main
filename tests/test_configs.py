"""Maintained config, legacy-profile and episodic-batching contracts."""
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


def test_sav_has_one_public_switch_set_and_retains_training_protocol():
    explicit = yaml.safe_load((CONFIGS / "sav.yaml").read_text())
    fs = explicit["FEW_SHOT"]
    assert fs["DECOUPLING"] == dict(QUERY_ENABLE=False, SUPPORT_ENABLE=False,
                                  M_ENABLE=False, RHO_ENABLE=False)
    assert not {"LABEL_PRE_TEMPORAL", "POT_ROUTE", "SUPPORT_TEXT_FUSION"} & fs.keys()
    assert "SHARED_QUERY_ENABLE" not in fs["TEXT_CONCAT_ST"]
    assert "SHARED_SUPPORT_ENABLE" not in fs["TEXT_CONCAT_ST"]
    assert set(fs["QUERY_CLASS_MATCHABILITY"]) == {"LEARNED_VERIFIER"}
    cfg, old = get_cfg(), get_cfg()
    cfg.merge_from_file(str(CONFIGS / "sav.yaml"))
    old.merge_from_file(str(CONFIGS / "sav_shared_support_query.yaml"))
    assert_and_infer_cfg(cfg)
    for name in ("DATA", "MODEL", "MF", "POINT_INFO", "SOLVER", "TRAIN", "TEST", "DATA_LOADER"):
        assert cfg[name] == old[name]
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST == old.FEW_SHOT.TEXT_CONCAT_ST
    assert cfg.FEW_SHOT.N_WAY == 5 and cfg.FEW_SHOT.K_SHOT == 1
    assert cfg.FEW_SHOT.CLASS_LOSS_LAMBDA == cfg.FEW_SHOT.Q2S_LOSS_LAMBDA == 1.
    assert cfg.OUTPUT_DIR == "output/sav"


@pytest.mark.parametrize("name", ["sav_text_concat_s1_base.yaml",
    "sav_text_concat_s1_m_rho_on.yaml", "sav_text_concat_shared_query.yaml",
    "sav_shared_support_query.yaml", "sav_label_pre_temporal.yaml", "tinyvirat.yaml"])
def test_legacy_profiles_do_not_opt_into_public_switches(name):
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIGS / name))
    before = cfg.clone()
    assert_and_infer_cfg(cfg)
    assert cfg == before
    assert all(v is None for v in cfg.FEW_SHOT.DECOUPLING.values())


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
