import json
from types import SimpleNamespace

import torch

from trokens.models.pointformer import Pointformer


def _model_with_psr_cfg():
    model = Pointformer.__new__(Pointformer)
    model.pot_route_cfg = SimpleNamespace(
        AFFINITY_TAU=0.07,
        ENTROPIC_EPS=0.01,
        MAX_ITERS=8,
        STOP_TOL=1e-5,
        UOT3D_ENTROPIC_EPS=0.03,
        UOT3D_MU_LOGIT_SCALE=5.0,
        UOT3D_RHO_FRAME=0.3,
        UOT3D_RHO_TRAJ=0.5,
        UOT3D_RHO_VIS=0.5,
        UOT3D_TARGET_MIX=0.85,
        UOT3D_TOTAL_MASS=1.0,
        UOT3D_TAU_FRAME=0.12,
        UOT3D_TAU_TRAJ=0.07,
        UOT3D_TAU_VIS=0.07,
        SHARED_TAU_LABEL=0.07,
        SHARED_THETA=0.2,
        SHARED_TAU_STRENGTH=0.1,
        UOT3D_SHARED_ENABLE=True,
        UOT3D_SHARED_COST_WEIGHT=0.5,
        UOT3D_SHARED_RATIO=0.2,
        UOT3D_VIS_PRIVATE_WEIGHT=1.0,
        UOT3D_VIS_SHARED_WEIGHT=1.0,
        DEBUG_TOPK=3,
        DEBUG_SAVE_TOP_TOKENS=True,
    )
    return model


def test_compute_sharedness_3d_reuses_entropy_strength_gate():
    model = _model_with_psr_cfg()
    sim = torch.tensor(
        [
            [[0.02, 0.90, 0.75, 0.80]],
            [[0.01, 0.10, 0.70, 0.78]],
            [[0.00, 0.00, 0.72, 0.79]],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor([[True, True, True, False]])

    sharedness, components = model._compute_sharedness_3d(
        sim,
        point_mask,
        return_components=True,
    )

    background = sharedness[0, 0]
    private = sharedness[0, 1]
    shared = sharedness[0, 2]

    assert shared > background
    assert shared > private
    assert sharedness[0, 3] == 0.0
    assert components["semantic_strength"][0, 0] < components["semantic_strength"][0, 2]
    assert components["label_entropy"][0, 1] < components["label_entropy"][0, 2]


def test_psr_3d_uot_transport_shapes_and_debug_are_stable():
    model = _model_with_psr_cfg()
    positive_text = torch.eye(4, dtype=torch.float32)[:3]
    st_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.8, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.9, 0.1, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    point_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, True],
        ]
    )
    support_global = st_tokens[point_mask].mean(dim=0)

    out = model._compute_avg_3d_uot_transport(
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        return_debug=True,
    )

    assert out["st_transport"].shape == (3, 2, 4)
    assert out["shared_transport"].shape == (3, 2, 4)
    assert out["cost_ext"].shape == (4, 2, 4)
    assert torch.isfinite(out["st_transport"]).all()
    assert torch.isfinite(out["shared_transport"]).all()
    assert out["st_transport"][:, 1, 2].sum() == 0.0
    assert out["shared_transport"][:, 1, 2].sum() == 0.0

    debug = out["debug"]
    assert debug["debug_type"] == "psr_3d_uot_soft"
    assert "transport_overlap" in debug
    assert "sharedness_summary" in debug
    assert "semantic_strength_summary" in debug
    assert "label_entropy_summary" in debug
    assert "shared_top_tokens" in debug
    assert "target_vs_shared_overlap" in debug["targets"][0]
    assert "shared_absorption_ratio" in debug["targets"][0]
    json.dumps(debug)