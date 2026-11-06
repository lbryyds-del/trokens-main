"""Shared DinoTxt interface: scale placement, gradients and checkpoint safety."""
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from trokens.config.defaults import get_cfg, assert_and_infer_cfg
from trokens.config.feature_scale import (
    feature_gain, scale_feature, validate_feature_scale_checkpoint,
)
from test_ablation_switches import _model, _episode, _loss


ROOT = Path(__file__).resolve().parents[1]
GAIN = 9.770436058356436


def test_disabled_and_unit_gains_are_exact_identity():
    cfg = get_cfg()
    cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = GAIN
    x = torch.randn(2, 4, requires_grad=True)
    before = x.detach().clone()
    assert scale_feature(x, feature_gain(cfg, "appearance")) is x
    assert scale_feature(x, 1.) is x
    y = scale_feature(x, 3.)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3.))
    torch.testing.assert_close(y, x * 3., rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["sav", "sav_rho_only", "sav_decoupled_base", "tinyvirat"])
def test_active_profiles_keep_unscaled_frozen_dinotxt(name):
    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / "configs/trokens" / f"{name}.yaml"))
    assert_and_infer_cfg(cfg)
    assert cfg.MODEL.FEAT_EXTRACTOR == "dinotxt_vitl14_reg4"
    assert not cfg.MODEL.TRAIN_BACKBONE and not cfg.MODEL.FEATURE_SCALE.ENABLE
    assert tuple(feature_gain(cfg, b) for b in ("appearance", "hod", "ctd")) == (1., 1., 1.)


@pytest.mark.parametrize("key,value", [
    ("APPEARANCE_GAIN", 0.), ("APPEARANCE_GAIN", -1.),
    ("APPEARANCE_GAIN", float("nan")), ("APPEARANCE_GAIN", float("inf")),
    ("HOD_GAIN", -1.), ("CTD_GAIN", float("inf")),
])
def test_invalid_gains_fail_early(key, value):
    cfg = get_cfg()
    cfg.MODEL.FEATURE_SCALE[key] = value
    with pytest.raises(ValueError, match="FEATURE_SCALE"):
        assert_and_infer_cfg(cfg)


@pytest.mark.parametrize("override", [
    ("FEAT_EXTRACTOR", "dino"), ("TRAIN_BACKBONE", True),
    ("APPEARANCE_MODULE_DISABLE", True),
])
def test_adapter_requires_frozen_enabled_dinotxt(override):
    cfg = get_cfg()
    cfg.MODEL.FEAT_EXTRACTOR = "dinotxt_vitl14_reg4"
    cfg.MODEL.FEATURE_SCALE.ENABLE = True
    cfg.MODEL[override[0]] = override[1]
    with pytest.raises(ValueError, match="FEATURE_SCALE"):
        assert_and_infer_cfg(cfg)


def test_real_forward_scales_before_positions_and_after_motion_modules():
    model = _model((False, False, False)).eval()
    cfg = model.cfg
    cfg.MF.USE_BASE_POS_EMBED = True
    cfg.MODEL.FEATURE_SCALE.ENABLE = True
    cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = 3.
    cfg.MODEL.FEATURE_SCALE.HOD_GAIN = .2
    cfg.MODEL.FEATURE_SCALE.CTD_GAIN = .4
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = True
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = True
    f, _, meta = _episode()
    before = f.clone()
    meta["hod_feat"] = torch.zeros(4, 5, 3, 2)
    position = nn.Parameter(torch.ones_like(f))
    hod = nn.Parameter(torch.full_like(f, 2.))
    ctd = nn.Parameter(torch.full_like(f, 4.))
    model.add_st_pos_embeddings = lambda x: x + position
    model.hod_motion_module = lambda x: hod
    model.cross_motion_module = lambda x, mask: ctd
    captured = []
    original = model.pt_forward
    model.pt_forward = lambda x, metadata: (captured.append(x), original(x, metadata))[1]
    model({"video": f, "metadata": meta})
    torch.testing.assert_close(captured[0], 3*f + position + .2*hod + .4*ctd)
    torch.testing.assert_close(f, before, rtol=0, atol=0)
    captured[0].sum().backward()
    torch.testing.assert_close(position.grad, torch.ones_like(position))
    torch.testing.assert_close(hod.grad, torch.full_like(hod, .2))
    torch.testing.assert_close(ctd.grad, torch.full_like(ctd, .4))


@pytest.mark.parametrize("flags", [
    (False, False, False), (True, True, False), (True, True, True),
])
def test_unit_gain_preserves_forward_and_checkpoint_keys(flags):
    model = _model(flags).eval()
    f, _, meta = _episode()
    with torch.no_grad():
        old = model({"video": f, "metadata": copy.deepcopy(meta)})
        keys = tuple(model.state_dict())
        model.cfg.MODEL.FEATURE_SCALE.ENABLE = True
        new = model({"video": f, "metadata": copy.deepcopy(meta)})
    if isinstance(old, tuple):
        for a, b in zip(old[:2], new[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        for name in ("global_classification_logits", "query_partial_q2s_logits"):
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
    assert tuple(model.state_dict()) == keys


def test_rho_path_no_longer_samples_raw_m_evidence(monkeypatch):
    model = _model((True, True, True)).eval()
    cfg = model.cfg
    cfg.MF.USE_BASE_POS_EMBED = True
    cfg.MODEL.FEATURE_SCALE.ENABLE = True
    cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = GAIN
    f, _, meta = _episode()
    position = torch.ones_like(f)
    model.add_st_pos_embeddings = lambda x: x + position
    captured = {}

    def capture(owner, fused, metadata, raw):
        captured.update(fused=fused, raw=raw)
        return captured

    monkeypatch.setattr("trokens.models.pointformer.text_concat_episode", capture)
    model({"video": f, "metadata": meta})
    assert captured["raw"] is None
    for role in (meta["support_mask"], ~meta["support_mask"]):
        torch.testing.assert_close(captured["fused"][role], GAIN*f[role] + position[role])


def test_real_motion_encoders_fusion_and_training_gradients():
    from trokens.models.branches.motion_blocks import HODMotionModule, CrossMotionModule

    model = _model((False, False, False)).train()
    cfg = model.cfg
    cfg.MODEL.FEATURE_SCALE.ENABLE = True
    cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = GAIN
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = True
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = True
    model.hod_motion_module = HODMotionModule(out_feature_dim=4, num_patches=5, in_feature_dim=2)
    model.cross_motion_module = CrossMotionModule(out_feature_dim=4, num_patches=5, in_fea_dim_crossmotion=10)
    f, _, meta = _episode()
    f.requires_grad_()
    meta["hod_feat"] = torch.randn(4, 5, 3, 2)
    meta["pred_tracks"] = torch.rand(4, 3, 5, 2) * 2 - 1
    branch_outputs, fused_inputs = {}, []
    model.hod_motion_module.register_forward_hook(
        lambda module, inputs, output: branch_outputs.update(hod=output))
    model.cross_motion_module.register_forward_hook(
        lambda module, inputs, output: branch_outputs.update(ctd=output))
    original = model.pt_forward
    model.pt_forward = lambda x, metadata: (fused_inputs.append(x), original(x, metadata))[1]
    out = model({"video": f, "metadata": meta})
    torch.testing.assert_close(fused_inputs[0], GAIN*f + branch_outputs["hod"] + branch_outputs["ctd"])
    loss, _, _ = _loss(model, out, meta)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(f.grad).all()
    assert f.grad[:2].abs().sum() > 0 and f.grad[2:].abs().sum() > 0
    for module in (model.hod_motion_module, model.cross_motion_module):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_checkpoint_guard_handles_legacy_and_explicit_initialization():
    model = _model((False, False, False))
    old_cfg = model.cfg.dump()
    validate_feature_scale_checkpoint(model, old_cfg)
    validate_feature_scale_checkpoint(model, None)
    validate_feature_scale_checkpoint(model, {"MODEL": {}})
    model.cfg.MODEL.FEATURE_SCALE.ENABLE = True
    model.cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = GAIN
    validate_feature_scale_checkpoint(model, model.cfg.dump())
    for saved in (old_cfg, None, {"MODEL": {}}):
        with pytest.raises(ValueError, match="checkpoint gains"):
            validate_feature_scale_checkpoint(model, saved)
        validate_feature_scale_checkpoint(model, saved, allow_change=True)
    calibrated = model.cfg.dump()
    model.cfg.MODEL.FEATURE_SCALE.ENABLE = False
    with pytest.raises(ValueError, match="checkpoint gains"):
        validate_feature_scale_checkpoint(model, calibrated)


@pytest.mark.parametrize("direction", ["old_to_scaled", "scaled_to_old"])
def test_production_checkpoint_loader_checks_before_loading(tmp_path, direction):
    from trokens.utils.checkpoint import load_checkpoint

    source, target = (_model((False, False, False)) for _ in range(2))
    scaled = source if direction == "scaled_to_old" else target
    scaled.cfg.MODEL.FEATURE_SCALE.ENABLE = True
    scaled.cfg.MODEL.FEATURE_SCALE.APPEARANCE_GAIN = GAIN
    with torch.no_grad():
        source.head.weight.add_(1.)
    path = tmp_path / "source.pyth"
    torch.save({"model_state": source.state_dict(), "epoch": 3,
                "cfg": source.cfg.dump()}, path)
    before = target.head.weight.detach().clone()
    with pytest.raises(ValueError, match="checkpoint gains"):
        load_checkpoint(str(path), target, data_parallel=False)
    torch.testing.assert_close(target.head.weight, before, rtol=0, atol=0)
    # Changing scale is allowed only as explicit initialization, not resume.
    assert load_checkpoint(str(path), target, data_parallel=False, epoch_reset=True) == -1
    target.cfg = source.cfg.clone()
    assert load_checkpoint(str(path), target, data_parallel=False) == 3
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["sav", "tinyvirat"])
def test_main_launcher_validates_scale_in_dry_run(name):
    command = [sys.executable, str(ROOT / "tools/run_local.py"),
               "--cfg", f"configs/trokens/{name}.yaml", "--gpus", "0,1", "--dry-run"]
    valid = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--nproc_per_node=2" in valid.stdout
    invalid = subprocess.run(command + ["--", "MODEL.FEATURE_SCALE.APPEARANCE_GAIN", "0.0"],
                             cwd=ROOT, capture_output=True, text=True)
    assert invalid.returncode != 0 and "FEATURE_SCALE" in invalid.stderr
