import json
from types import SimpleNamespace

import torch

from trokens.models.pointformer import Pointformer


class _ConstantCMWCostNet:
    def __init__(self, reliability=0.37, num_families=5):
        self.reliability = float(reliability)
        self.num_families = int(num_families)

    def __call__(
        self,
        token_evidence,
        label_context,
        point_mask,
        min_reliability=0.02,
    ):
        del label_context
        reliability = token_evidence.new_full(
            token_evidence.shape[:3],
            max(self.reliability, float(min_reliability)),
        )
        reliability = reliability * point_mask.unsqueeze(0).to(reliability.dtype)
        family_prob = token_evidence.new_full(
            (token_evidence.shape[0], self.num_families),
            1.0 / float(self.num_families),
        )
        candidate_reliability = reliability.unsqueeze(-1).expand(
            *reliability.shape,
            self.num_families,
        )
        return reliability, {
            "candidate_reliability": candidate_reliability,
            "family_prob": family_prob,
        }


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
        UOT3D_SHARED_RATIO=0.2,
        UOT3D_VIS_PRIVATE_WEIGHT=1.0,
        UOT3D_VIS_SHARED_WEIGHT=1.0,
        CMW_COST_NUM_FAMILIES=5,
        CMW_COST_HIDDEN_DIM=128,
        CMW_COST_MIN_RELIABILITY=0.02,
        CMW_COST_MARGIN_TAU=0.1,
        DEBUG_TOPK=3,
        DEBUG_SAVE_TOP_TOKENS=True,
    )
    model.cmw_cost_net = _ConstantCMWCostNet(reliability=0.37, num_families=5)
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


def test_cmw_replaces_only_private_cost_rows():
    model = _model_with_psr_cfg()
    model.cmw_cost_net = _ConstantCMWCostNet(reliability=0.37, num_families=5)

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
        target_label_indices=torch.tensor([0, 2]),
    )

    valid_private = point_mask.unsqueeze(0).expand(3, 2, 4)
    expected_private_cost = torch.full_like(out["private_cost"], 0.63)
    assert torch.allclose(
        out["private_cost"][valid_private],
        expected_private_cost[valid_private],
        atol=1e-6,
    )
    assert torch.allclose(
        out["cost"][valid_private],
        expected_private_cost[valid_private],
        atol=1e-6,
    )
    assert out["private_cost"][:, 1, 2].min() == 1e4
    assert out["cost_ext"][:3, 1, 2].min() == 1e4
    assert torch.allclose(
        out["cost_ext"][-1][point_mask],
        1.0 - out["sharedness"][point_mask],
        atol=1e-6,
    )
    assert out["st_transport"].shape == (2, 2, 4)
    assert out["target_label_indices"].tolist() == [0, 2]

    debug = out["debug"]
    assert debug["cost_source"] == "cmw_private_cost"
    assert "cmw_private_reliability_summary" in debug
    assert "cmw_evidence_mean" in debug
    assert "cmw_target_reliability_summary" in debug["targets"][0]
    assert "cmw_target_top_reliable_tokens" not in debug["targets"][0]
    json.dumps(debug)


def test_cmw_raw_label_axis_keeps_target_subset_outputs():
    model = _model_with_psr_cfg()
    model.cfg = SimpleNamespace(POINT_INFO=SimpleNamespace(USE_PT_QUERY_MASK=False))
    model.num_classes = 5
    model.atomic_label_names = [str(class_id) for class_id in range(5)]
    model._should_log_pot_debug = lambda: False
    model._get_pot_label_text_features = lambda ids, dtype: torch.nn.functional.one_hot(
        ids.cpu(),
        num_classes=5,
    ).to(dtype=dtype)

    calls = []

    def fake_uot(
        st_tokens,
        point_mask,
        support_global,
        positive_text,
        **kwargs,
    ):
        del support_global
        calls.append((positive_text.clone(), kwargs["target_label_indices"].clone()))
        num_targets = int(kwargs["target_label_indices"].numel())
        return {
            "st_transport": torch.ones(num_targets, *point_mask.shape),
            "sim": torch.zeros(positive_text.shape[0], *point_mask.shape),
        }

    model._compute_avg_3d_uot_transport = fake_uot
    value_tokens = torch.randn(2, 2, 3, 5)
    metadata = {
        "support_mask": torch.tensor([True, False]),
        "episode_positive_labels": torch.tensor(
            [[1, 0, 1], [0, 0, 0]],
            dtype=torch.float32,
        ),
        "raw_positive_labels": torch.tensor(
            [[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        ),
        "episode_class_ids": torch.tensor([0, 2, 3]),
        "pred_visibility": torch.ones(2, 2, 3, dtype=torch.bool),
    }

    aux = model._build_pot_support_prototypes(
        None,
        None,
        None,
        value_tokens,
        metadata,
    )
    positive_text, target_label_indices = calls[0]
    assert positive_text.shape[0] == 3
    assert target_label_indices.detach().cpu().tolist() == [0, 2]
    assert aux["support_branch_class_indices"].detach().cpu().tolist() == [0, 2]
    assert aux["support_conditioned_patch_tokens"].shape[0] == 2
