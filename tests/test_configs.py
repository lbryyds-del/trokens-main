"""Active config contracts for the consolidated SAV rho/change model."""
import itertools
from pathlib import Path

import pytest
import torch

from trokens.config.defaults import assert_and_infer_cfg, get_cfg

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "trokens"


def config(name):
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIGS / name))
    return assert_and_infer_cfg(cfg)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_dataset_config_merges(path):
    config(path.name)


def test_sav_final_model_and_training_protocol():
    cfg = config("sav.yaml")
    fs = cfg.FEW_SHOT
    assert dict(fs.DECOUPLING) == dict(QUERY_ENABLE=True, SUPPORT_ENABLE=True, RHO_ENABLE=True)
    assert fs.ENTITY_CHANGE.ENABLE and fs.ENTITY_CHANGE.STATE_ENABLE and fs.ENTITY_CHANGE.GEO_ENABLE
    assert fs.ENTITY_CHANGE.DETACH_INPUTS and fs.ENTITY_CHANGE.LR_MULT == 100.
    assert fs.ENTITY_CHANGE.LAMBDA_STATE_INIT == fs.ENTITY_CHANGE.LAMBDA_GEO_INIT == 0.
    assert (cfg.RNG_SEED, cfg.NUM_GPUS, fs.N_WAY, fs.K_SHOT) == (1, 2, 5, 1)
    assert (fs.TRAIN_EPISODES, fs.TEST_EPISODES, fs.TRAIN_QUERY_PER_CLASS) == (2000, 10000, 6)
    assert (cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_EPOCH, cfg.SOLVER.WARMUP_EPOCHS) == (3e-5, 10, 3)
    assert cfg.SOLVER.GRAD_ACCUM_STEPS == 2 and cfg.SOLVER.LR_POLICY == "cosine"
    assert not cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE
    assert not cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE
    assert not cfg.POINT_INFO.HOD.GET_FEAT
    assert not cfg.MODEL.FEATURE_SCALE.ENABLE and not cfg.MODEL.TRAIN_BACKBONE
    assert not cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.CHECKPOINT_FILE_PATH == ""
    assert cfg.TRAIN.CHECKPOINT_KEEP_ALL and cfg.TEST.TEST_EPOCH_NUM == 10
    assert cfg.TEST.DATA_CSV == "" and not cfg.TEST.DROP_EMPTY_QUERY_ROWS
    assert Path(cfg.TRAIN.EVAL_DATA_CSV).is_file()
    assert cfg.OUTPUT_DIR == "output/sav/rho_change"


def test_main_matches_trained_pilot_except_output_and_final_test_pool():
    main, pilot = config("sav.yaml"), config("sav_entity_change_s1.yaml")
    for field in ("FEW_SHOT", "SOLVER", "TRAIN", "DATA", "MODEL", "MF", "POINT_INFO", "DATA_LOADER"):
        assert main[field] == pilot[field], field
    assert main.TEST.DATA_CSV == ""
    assert pilot.TEST.DATA_CSV.endswith("/reserved.csv")
    main.TEST.DATA_CSV = pilot.TEST.DATA_CSV
    main.OUTPUT_DIR = pilot.OUTPUT_DIR
    main.WANDB.EXP_NAME = pilot.WANDB.EXP_NAME
    assert main == pilot


def test_baselines_change_only_the_declared_evidence_switches():
    full, rho, base = [config(n) for n in ("sav.yaml", "sav_rho_only.yaml", "sav_decoupled_base.yaml")]
    assert not rho.FEW_SHOT.ENTITY_CHANGE.ENABLE and rho.FEW_SHOT.DECOUPLING.RHO_ENABLE
    assert not base.FEW_SHOT.ENTITY_CHANGE.ENABLE and not base.FEW_SHOT.DECOUPLING.RHO_ENABLE
    for cfg in (full, rho, base):
        assert cfg.FEW_SHOT.DECOUPLING.QUERY_ENABLE and cfg.FEW_SHOT.DECOUPLING.SUPPORT_ENABLE
        assert cfg.SOLVER == full.SOLVER and cfg.TRAIN == full.TRAIN and cfg.TEST == full.TEST


def test_retired_experiment_options_cannot_be_enabled():
    cfg = get_cfg()
    assert "M_ENABLE" not in cfg.FEW_SHOT.DECOUPLING
    assert "PAIR_COMPETITION" not in cfg.FEW_SHOT
    verifier = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER
    assert "RHO_QUERY_FEEDBACK" not in verifier and "QUALITY_MODE" not in verifier
    assert "QUALITY_ENABLE" not in cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert "SEMANTIC_MIL_WEIGHT" not in verifier and "M_AUX_WEIGHT" not in cfg.FEW_SHOT.MATCHING
    for key in ("FEW_SHOT.DECOUPLING.M_ENABLE", "FEW_SHOT.PAIR_COMPETITION.ENABLE",
                "FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.RHO_QUERY_FEEDBACK"):
        with pytest.raises((KeyError, AssertionError)):
            cfg.clone().merge_from_list([key, "True"])


def test_consolidated_model_keeps_trained_checkpoint_identifiers():
    from trokens.models.text_concat_conditioning import TextConcatST
    from trokens.models.query_evidence_verifier import QueryEvidenceVerifier
    from trokens.models.entity_change_verifier import EntityChangeVerifier

    cfg = config("sav.yaml")
    concat = TextConcatST(cfg.FEW_SHOT.TEXT_CONCAT_ST, uniqueness=True,
                          entity_change=cfg.FEW_SHOT.ENTITY_CHANGE)
    assert concat.architecture_signature.tolist() == [
        2, 1, 1, 1, 0, 1, 9, 1, 1, 1, 1, 0, 0, 16, 4, 4, 16, 3]
    rho = QueryEvidenceVerifier(4, cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER)
    assert rho.architecture_version.item() == 2
    assert set(rho.state_dict()) == {
        "architecture_version", "metric_temperature_raw", "metric_scale_raw"}
    change = EntityChangeVerifier(cfg.FEW_SHOT.ENTITY_CHANGE)
    assert set(dict(change.named_parameters())) == {
        "lambda_state", "lambda_geo", "state_temperature_raw", "state_scale_raw",
        "geo_temperature_raw", "geo_scale_raw"}


def test_internal_matching_defaults_preserve_explicit_profile_behavior():
    defaults = config("sav_rho_only.yaml")
    explicit = defaults.clone()
    explicit.FEW_SHOT.MATCHING.merge_from_other_cfg(type(explicit.FEW_SHOT.MATCHING)({
        "LOGIT_SCALE": 10.0, "LOGIT_BIAS": -2.0, "BASE_LOSS_WEIGHT": 0.6,
        "FINAL_LOSS_WEIGHT": 0.4, "RHO_AUX_WEIGHT": 0.05, "TRANSPORT_STRENGTH": 1.0,
    }))
    for flags in itertools.product((False, True), repeat=3):
        if not flags[0] and flags[2]:
            continue
        actual, expected = defaults.clone(), explicit.clone()
        for key, value in zip(("QUERY_ENABLE", "SUPPORT_ENABLE", "RHO_ENABLE"), flags):
            actual.FEW_SHOT.DECOUPLING[key] = value
            expected.FEW_SHOT.DECOUPLING[key] = value
        assert_and_infer_cfg(actual)
        assert_and_infer_cfg(expected)
        assert actual == expected


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
