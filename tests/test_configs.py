"""Maintained config, legacy-profile and episodic-batching contracts."""
import itertools
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


@pytest.mark.parametrize("name", ["sav", "tinyvirat"])
def test_main_profiles_enable_motion_without_text_or_judgments(name):
    explicit = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    fs = explicit["FEW_SHOT"]
    assert fs["DECOUPLING"] == dict(QUERY_ENABLE=False, SUPPORT_ENABLE=False,
                                  M_ENABLE=False, RHO_ENABLE=False)
    assert not {"LABEL_PRE_TEMPORAL", "POT_ROUTE", "SUPPORT_TEXT_FUSION"} & fs.keys()
    assert "SHARED_QUERY_ENABLE" not in fs["TEXT_CONCAT_ST"]
    assert "SHARED_SUPPORT_ENABLE" not in fs["TEXT_CONCAT_ST"]
    assert set(fs["QUERY_CLASS_MATCHABILITY"]) == {"LEARNED_VERIFIER"}
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIGS / f"{name}.yaml"))
    assert_and_infer_cfg(cfg)
    assert 'MATCHING' not in fs
    concat = cfg.FEW_SHOT.TEXT_CONCAT_ST
    assert not concat.ENABLE
    assert concat.SHARED_QUERY_ENABLE and concat.SHARED_SUPPORT_ENABLE
    assert concat.TEMPORAL_TEXT_TOKENS == concat.SPATIAL_TEXT_TOKENS == 1
    route = cfg.FEW_SHOT.POT_ROUTE
    assert not route.ENABLE and not route.QUERY_PARTIAL_ENABLE
    assert (route.QUERY_PARTIAL_LOGIT_ALPHA, route.QUERY_PARTIAL_LOGIT_BIAS) == (1., 0.)
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert not match.ENABLE and not match.QUALITY_ENABLE and not match.UNIQUENESS_ENABLE
    assert not match.DUAL_LOGIT_LOSS_ENABLE and not match.BASE_ONLY_LOSS_ENABLE
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == (1., 0.)
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == 0.
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == 0.
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == 0.
    assert cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE
    assert cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE
    assert not cfg.MODEL.APPEARANCE_MODULE_DISABLE
    assert cfg.FEW_SHOT.N_WAY == 5
    assert type(cfg.FEW_SHOT.K_SHOT) is int and cfg.FEW_SHOT.K_SHOT > 0
    assert cfg.FEW_SHOT.K_SHOT == fs['K_SHOT']
    assert cfg.FEW_SHOT.TRAIN_EPISODES == 2000
    assert cfg.FEW_SHOT.TEST_EPISODES == 10000
    assert cfg.SOLVER.MAX_EPOCH == 20
    assert cfg.FEW_SHOT.CLASS_LOSS_LAMBDA == cfg.FEW_SHOT.Q2S_LOSS_LAMBDA == 1.
    assert cfg.OUTPUT_DIR == f"output/{name}/dinotxt_scaled"


@pytest.mark.parametrize("name", ["sav", "tinyvirat"])
def test_internal_matching_defaults_preserve_explicit_profile_behavior(name):
    defaults = get_cfg()
    defaults.merge_from_file(str(CONFIGS / f"{name}.yaml"))
    explicit = defaults.clone()
    explicit.FEW_SHOT.MATCHING.merge_from_other_cfg(type(explicit.FEW_SHOT.MATCHING)({
        "LOGIT_SCALE": 10.0,
        "LOGIT_BIAS": -2.0,
        "BASE_LOSS_WEIGHT": 0.6,
        "FINAL_LOSS_WEIGHT": 0.4,
        "M_AUX_WEIGHT": 0.05,
        "RHO_AUX_WEIGHT": 0.05,
        "TRANSPORT_STRENGTH": 1.0,
    }))
    for flags in itertools.product((False, True), repeat=4):
        if not flags[0] and any(flags[2:]):
            continue
        actual, expected = defaults.clone(), explicit.clone()
        for key, value in zip(("QUERY_ENABLE", "SUPPORT_ENABLE", "M_ENABLE", "RHO_ENABLE"), flags):
            actual.FEW_SHOT.DECOUPLING[key] = value
            expected.FEW_SHOT.DECOUPLING[key] = value
        assert_and_infer_cfg(actual)
        assert_and_infer_cfg(expected)
        assert actual == expected


def test_sav_retains_historical_training_protocol():
    cfg, old = get_cfg(), get_cfg()
    cfg.merge_from_file(str(CONFIGS / "sav.yaml"))
    old.merge_from_file(str(CONFIGS / "sav_shared_support_query.yaml"))
    assert_and_infer_cfg(cfg)
    # The interface gain is the only intentional model change from this
    # historical profile; dataset, optimizer, shots and training stay intact.
    assert cfg.MODEL.FEATURE_SCALE.ENABLE
    assert not old.MODEL.FEATURE_SCALE.ENABLE
    old.MODEL.FEATURE_SCALE = cfg.MODEL.FEATURE_SCALE.clone()
    for name in ("DATA", "MODEL", "MF", "POINT_INFO", "SOLVER", "TRAIN", "TEST", "DATA_LOADER"):
        assert cfg[name] == old[name]


def test_tiny_and_sav_only_differ_in_dataset_metadata_or_shot_count():
    sav, tiny = get_cfg(), get_cfg()
    sav.merge_from_file(str(CONFIGS / "sav.yaml"))
    tiny.merge_from_file(str(CONFIGS / "tinyvirat.yaml"))
    assert_and_infer_cfg(sav)
    assert_and_infer_cfg(tiny)

    def differences(left, right, prefix=""):
        assert left.keys() == right.keys()
        result = set()
        for key in left:
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(left[key], dict):
                result.update(differences(left[key], right[key], path))
            elif left[key] != right[key]:
                result.add(path)
        return result

    # Shot count is an independent per-dataset experiment choice.
    assert differences(sav, tiny) - {"FEW_SHOT.K_SHOT"} == {
        "OUTPUT_DIR", "TRAIN.DATASET", "TRAIN.CLASS_NAME", "TEST.DATASET",
        "TEST.CLASS_NAME", "TEST.SEEN_LABELS", "TEST.NOVEL_LABELS",
        "TEST.DROP_EMPTY_QUERY_ROWS", "MODEL.NUM_CLASSES", "DATA.LABEL_PROMPTS",
        "DATA.PATH_TO_DATA_DIR", "DATA.PATH_TO_SPLIT_DIR", "DATA.PATH_TO_TROKEN_PT_DATA",
    }
    assert not sav.TEST.DROP_EMPTY_QUERY_ROWS
    assert tiny.TEST.DROP_EMPTY_QUERY_ROWS


@pytest.mark.parametrize("name", ["sav_text_concat_s1_base.yaml",
    "sav_text_concat_s1_m_rho_on.yaml", "sav_text_concat_shared_query.yaml",
    "sav_shared_support_query.yaml", "sav_label_pre_temporal.yaml"])
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
