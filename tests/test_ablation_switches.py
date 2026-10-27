"""Compact maintained regression for the public SAV ablation switches."""
import copy
import itertools
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trokens.config.defaults import get_cfg
from trokens.config.ablation import resolve_ablation_cfg
from trokens.models.pointformer import Pointformer
from trokens.models.attention import TrajectoryAttentionBlock
from trokens.models.text_concat_conditioning import text_concat_episode
from trokens.utils.checkpoint import validate_label_pre_temporal_checkpoint
from tools.few_shot_multilabel import (compute_query_partial_q2s_loss,
    compute_learned_evidence_mil_loss, support_query_split_multilabel)


CONFIG = Path(__file__).resolve().parents[1] / 'configs/trokens/sav.yaml'
sys.path.insert(0, str(CONFIG.parents[2] / 'tools'))
COMBINATIONS = list(itertools.product((False, True), repeat=4))  # Query, Support, m, rho
VALID = [x for x in COMBINATIONS if x[0] or not any(x[2:])]
TEXT_VALID = [x for x in VALID if any(x[:2])]


def _cfg(flags):
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG))
    for name, value in zip(('QUERY_ENABLE', 'SUPPORT_ENABLE', 'M_ENABLE', 'RHO_ENABLE'), flags):
        cfg.FEW_SHOT.DECOUPLING[name] = value
    return resolve_ablation_cfg(cfg)


def _model(flags, mode='selective', archived_shared=False):
    """Real encoder/verifier/head, with small tensors and no backbone downloads."""
    torch.manual_seed(71)
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = _cfg(flags)
    cfg = model.cfg
    if archived_shared:
        # Explicit old v5 profile, not the new public four-switch baseline.
        for name in cfg.FEW_SHOT.DECOUPLING:
            cfg.FEW_SHOT.DECOUPLING[name] = None
        cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE = True
        cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE = True
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = True
        cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE = True
        cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA = 10.
        cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS = -2.
    cfg.TEST.SEEN_LABELS = [0, 1]
    cfg.MODEL.NUM_CLASSES = 2
    cfg.FEW_SHOT.TEXT_CONCAT_ST.CLASS_CHUNK_SIZE = 1
    cfg.FEW_SHOT.TEXT_CONCAT_ST.VIDEO_CHUNK_SIZE = 1
    cfg.FEW_SHOT.TEXT_CONCAT_ST.ACTIVATION_CHECKPOINT = mode == 'full'
    cfg.FEW_SHOT.TEXT_CONCAT_ST.MLP_CHECKPOINT = mode == 'selective'
    cfg.FEW_SHOT.TEXT_CONCAT_ST.READOUT_CHECKPOINT = mode == 'selective'
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.QUALITY_HIDDEN_DIM = 8
    cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.QUERY_CHUNK_SIZE = 1
    model.embed_dim, model.depth, model.point_grid_size, model.temporal_resolution = 4, 1, 1, 3
    model.pot_route_cfg = cfg.FEW_SHOT.POT_ROUTE
    model.use_frame_softmax_route = model.use_support_text_fusion = model.use_label_pre_temporal = False
    model.is_multilabel_few_shot = True
    model.use_query_partial_q2s = cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE
    model.use_text_concat_st = cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE
    model.feat_extractor_type = 'dinotxt_vitl14_reg4'
    model.cls_token = nn.Parameter(torch.randn(1, 1, 4))
    model.pos_drop, model.head_drop = nn.Identity(), nn.Identity()
    model.blocks = nn.ModuleList([TrajectoryAttentionBlock(
        cfg=cfg, dim=4, num_heads=2, mlp_ratio=2., qkv_bias=True, use_pt_visibility=True)])
    model.norm, model.head, model.pre_logits = nn.LayerNorm(4), nn.Linear(4, 2), nn.Linear(4, 4)
    model._get_pot_label_text_features = lambda ids, dtype: torch.eye(4, dtype=dtype, device=ids.device)[ids]
    model._init_query_evidence_verifier()
    model._init_text_concat_st()
    # Feed synthetic trajectory features into the REAL default forward entry.
    # Only backbone/sampling/motion are stubbed; pt_forward/readout are intact.
    cfg.MF.USE_BASE_POS_EMBED = False
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = False
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = False
    model.get_dinotxt_features = lambda video: video
    model._sample_point_features = lambda features, tracks, **kwargs: features
    return model


def _episode():
    rng = torch.Generator().manual_seed(414)
    f, raw = [torch.randn(4, 3, 5, 4, generator=rng) for _ in range(2)]
    mask = torch.ones(4, 3, 5).bool()
    visibility = mask.clone()
    visibility[0, 1, 2:] = False
    visibility[2, 1] = False
    return f, raw, dict(support_mask=torch.tensor([True, True, False, False]),
        sample_type=['support', 'support', 'query', 'query'], pred_tracks=torch.zeros(4, 3, 5, 2),
        pred_query_mask=mask, pred_visibility=visibility, episode_class_ids=torch.tensor([0, 1]),
        episode_positive_labels=torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]).bool())


def _forward(model, f, raw, meta):
    if model.use_text_concat_st:
        return text_concat_episode(model, f, meta, raw)
    return model({'video': f, 'metadata': meta})


def _loss(model, out, meta):
    targets = meta['episode_positive_labels'][~meta['support_mask']].float()
    if isinstance(out, tuple):
        from train_few_shot import process_patch_tokens
        global_logits, patches, aux = out
        split = support_query_split_multilabel(patches, meta['episode_positive_labels'].float(), meta)
        scores = process_patch_tokens(model.cfg, split['support_preds'], split['query_preds'])
    else:
        global_logits, scores, aux = out['global_classification_logits'], out['query_partial_q2s_logits'], out
    q2s, _, metrics = compute_query_partial_q2s_loss(scores, targets, aux, model.cfg)
    total = q2s + F.binary_cross_entropy_with_logits(global_logits,
                                                   meta['episode_positive_labels'].float())
    return total, q2s, metrics


@pytest.mark.parametrize('flags', COMBINATIONS)
def test_config_dependencies_weights_and_idempotence(flags):
    query, support, m, rho = flags
    if not query and (m or rho):
        with pytest.raises(ValueError, match='require QUERY_ENABLE=True'):
            _cfg(flags)
        return
    cfg = _cfg(flags)
    expected = cfg.clone()
    cfg.freeze()
    resolve_ablation_cfg(cfg)
    assert cfg == expected and cfg.is_frozen()
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE == (query or support)
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE == (query or support)
    assert match.ENABLE == (query or support)
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.SHARED_QUERY_ENABLE == (not query)
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.SHARED_SUPPORT_ENABLE == (not support)
    assert (match.QUALITY_ENABLE, match.UNIQUENESS_ENABLE) == (m, rho)
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == ((.6, .4) if m or rho else (1., 0.))
    assert match.BASE_ONLY_LOSS_ENABLE == ((query or support) and not (m or rho))
    assert match.DUAL_LOGIT_LOSS_ENABLE == (m or rho)
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == (.05 if m else 0.)
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == (.05 if rho else 0.)
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == (1. if m or rho else 0.)


@pytest.mark.parametrize('flags', TEXT_VALID)
@pytest.mark.parametrize('mode', ['none', 'full', 'selective'])
def test_roles_matching_and_exact_loss_gradients(flags, mode, monkeypatch):
    query, support, m, rho = flags
    model = _model(flags, mode).train()
    f, raw, meta = _episode()
    f.requires_grad_()
    counts = {'ordinary': 0, 'text': 0}
    block = model.blocks[0]
    def counted(original, key):
        def forward(tokens, *args, **kwargs):
            counts[key] += tokens.shape[0]
            return original(tokens, *args, **kwargs)
        return forward
    monkeypatch.setattr(block, 'forward', counted(block.forward, 'ordinary'))
    monkeypatch.setattr(block, 'forward_text_concat', counted(block.forward_text_concat, 'text'))
    if not query and not support:
        def forbidden(*args):
            raise AssertionError('Both roles off must never encode text')
        monkeypatch.setattr(model, '_get_pot_label_text_features', forbidden)
    out = text_concat_episode(model, f, meta, raw)
    assert counts == {'ordinary': 2 * (not query) + 2 * (not support), 'text': 4 * query + 4 * support}
    qp, sp = out['query_partial_query_prototypes'], out['query_partial_support_prototypes']
    assert qp.shape == ((2, 2, 3, 4) if query else (2, 3, 4))
    assert sp.shape == (2, 3, 4)
    q, s = F.normalize(qp, dim=-1), F.normalize(sp, dim=-1)
    scores = torch.stack([torch.stack([
        .5 * ((a @ b.T).max(1).values.mean() + (a @ b.T).max(0).values.mean())
        for a, b in zip(q[row] if query else q[row:row+1].expand(2, -1, -1), s)]) for row in range(2)])
    torch.testing.assert_close(out['query_partial_q2s_base_logits'], 10 * scores - 2)
    assert ('query_quality_logits' in out) == m
    assert ('query_relation_logits' in out) == rho
    assert (model.query_evidence_verifier is not None) == (m or rho)
    total, actual, metrics = _loss(model, out, meta)
    labels = meta['episode_positive_labels'][2:].float()
    base = F.binary_cross_entropy_with_logits(out['query_partial_q2s_base_logits'], labels)
    expected = base
    if m or rho:
        final = F.binary_cross_entropy_with_logits(out['query_partial_q2s_logits'], labels)
        expected = .6 * base + .4 * final
        mass, relation = out.get('query_frame_absolute_patch_mass', 1.), out.get('query_frame_matchability', 1.)
        torch.testing.assert_close(out['query_frame_target_mass'], mass * relation)
    else:
        assert out['query_partial_q2s_logits'] is out['query_partial_q2s_base_logits']
        assert metrics['q2s_verified_loss'] == 0
    for name, enabled in [('quality', m), ('relation', rho)]:
        assert (f'q2s_{name}_mil_loss' in metrics) == enabled
        if enabled:
            expected = expected + .05 * compute_learned_evidence_mil_loss(
                out[f'query_{name}_logits'], out[f'query_{name}_valid'], labels)['loss']
    torch.testing.assert_close(actual, expected)
    total.backward()
    assert torch.isfinite(f.grad).all() and f.grad[:2].abs().sum() > 0 and f.grad[2:].abs().sum() > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('flags', TEXT_VALID)
def test_no_query_label_leakage_and_shared_role_ordinary_oracle(flags):
    model = _model(flags).eval()
    f, raw, meta = _episode()
    out = text_concat_episode(model, f, meta, raw)
    changed = copy.deepcopy(meta)
    changed['episode_positive_labels'][2:] = ~changed['episode_positive_labels'][2:]
    other = text_concat_episode(model, f, changed, raw)
    for key in ('query_partial_q2s_logits', 'global_classification_logits'):
        torch.testing.assert_close(out[key], other[key], rtol=0, atol=0)
    cls, patches = model.pt_forward(f, meta)
    if not flags[0]:
        torch.testing.assert_close(out['query_partial_query_prototypes'], patches[2:].mean(2))
        torch.testing.assert_close(out['global_classification_logits'][2:], model.head(cls[2:]))
    if not flags[1]:
        torch.testing.assert_close(out['query_partial_support_prototypes'], patches[:2].mean(2))
        torch.testing.assert_close(out['global_classification_logits'][:2], model.head(cls[:2]))


@pytest.mark.parametrize('flags', TEXT_VALID)
def test_empty_masks_unknown_labels_and_checkpoint_isolation(flags):
    model = _model(flags).eval()
    f, raw, meta = _episode()
    meta['pred_query_mask'].zero_()
    meta['episode_label_known_mask'] = torch.zeros(4, 2).bool()
    out = text_concat_episode(model, f, meta, raw)
    assert out['query_partial_support_prototypes'].eq(0).all()
    assert out['query_partial_query_prototypes'].eq(0).all()
    assert torch.isfinite(out['query_partial_q2s_logits']).all()
    validate_label_pre_temporal_checkpoint(model, model.state_dict())
    for other_flags in VALID:
        if flags != other_flags:
            with pytest.raises(ValueError):
                validate_label_pre_temporal_checkpoint(model, _model(other_flags).state_dict())


@pytest.mark.parametrize('query,support', itertools.product((False, True), repeat=2))
def test_five_way_matching_shape(query, support):
    model = _model((query, support, query, query)).train()
    model.head = nn.Linear(4, 5)
    model._get_pot_label_text_features = lambda ids, dtype: F.one_hot(ids % 4, 4).to(dtype)
    f, raw = torch.randn(7, 3, 5, 4, requires_grad=True), torch.randn(7, 3, 5, 4)
    meta = dict(support_mask=torch.tensor([True] * 5 + [False] * 2),
        sample_type=['support'] * 5 + ['query'] * 2, pred_tracks=torch.zeros(7, 3, 5, 2),
        pred_query_mask=torch.ones(7, 3, 5).bool(), pred_visibility=torch.ones(7, 3, 5).bool(),
        episode_class_ids=torch.arange(5), episode_positive_labels=torch.eye(5)[[0, 1, 2, 3, 4, 0, 3]].bool())
    out = _forward(model, f, raw, meta)
    if not query and not support:
        from train_few_shot import process_patch_tokens
        assert out[1].shape == (7, 3, 5, 4) and out[2] is None
        split = support_query_split_multilabel(out[1], meta['episode_positive_labels'].float(), meta)
        assert process_patch_tokens(model.cfg, split['support_preds'], split['query_preds']).shape == (2, 5)
        _loss(model, out, meta)[0].backward()
        assert torch.isfinite(f.grad).all()
        return
    assert out['query_partial_query_prototypes'].shape == ((2, 5, 3, 4) if query else (2, 3, 4))
    assert out['query_partial_support_prototypes'].shape == (5, 3, 4)
    assert out['query_partial_q2s_logits'].shape == (2, 5)
    _loss(model, out, meta)[0].backward()
    assert torch.isfinite(f.grad).all()


def test_invalid_switches_fail_before_model_loading_or_output_creation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from trokens.utils import parser
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG))
    cfg.FEW_SHOT.DECOUPLING.M_ENABLE = True
    with pytest.raises(ValueError, match='require QUERY_ENABLE=True'):
        Pointformer(cfg)  # validation precedes frozen-backbone initialization
    def forbidden(*args):
        raise AssertionError('Invalid configuration must not create checkpoints')
    monkeypatch.setattr(parser.cu, 'make_checkpoint_dir', forbidden)
    args = SimpleNamespace(cfg_file=str(CONFIG), output_dir=str(tmp_path / 'absent'),
                           opts=['FEW_SHOT.DECOUPLING.M_ENABLE', 'True'])
    with pytest.raises(ValueError, match='require QUERY_ENABLE=True'):
        parser.load_config(args)
    assert not (tmp_path / 'absent').exists()


def test_toggling_back_off_clears_every_derived_loss_and_transport():
    cfg = _cfg((True, True, True, True))
    cfg.FEW_SHOT.DECOUPLING.M_ENABLE = False
    cfg.FEW_SHOT.DECOUPLING.RHO_ENABLE = False
    resolve_ablation_cfg(cfg)
    match = cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    assert match.BASE_ONLY_LOSS_ENABLE and not match.DUAL_LOGIT_LOSS_ENABLE
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == (1., 0.)
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == 0.
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == 0.


def test_both_roles_off_automatically_restore_default_branch_and_toggle_back():
    cfg = _cfg((True, True, True, True))
    for name in cfg.FEW_SHOT.DECOUPLING:
        cfg.FEW_SHOT.DECOUPLING[name] = False
    resolve_ablation_cfg(cfg)
    expected = cfg.clone()
    cfg.freeze()
    resolve_ablation_cfg(cfg)
    assert cfg == expected and cfg.is_frozen()
    assert not cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE
    assert not cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE
    assert not cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE
    assert not cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE
    assert not cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA == 1.
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS == 0.
    cfg.defrost()
    cfg.FEW_SHOT.DECOUPLING.SUPPORT_ENABLE = True
    resolve_ablation_cfg(cfg)
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.ENABLE
    assert cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_ALPHA == 10.
    assert cfg.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_LOGIT_BIAS == -2.


@pytest.mark.parametrize('mode', ['none', 'full', 'selective'])
def test_default_branch_reuses_old_train_test_formula_and_gradients(mode, monkeypatch):
    # Import the production scorers, not a second implementation of the route.
    monkeypatch.syspath_prepend(str(CONFIG.parents[2] / 'tools'))
    from train_few_shot import process_patch_tokens as train_score
    from test_few_shot import process_patch_tokens as test_score
    from tools.few_shot_multilabel import support_query_split_multilabel, unpack_episode_output
    flags = (False, False, False, False)
    model = _model(flags, mode).train()
    f, raw, meta = _episode()
    f.requires_grad_()
    # Class zero has two Support videos: averaging must precede ReLU.
    meta['episode_positive_labels'][1] = torch.tensor([True, True])
    meta['sample_type'] = ['support', 'support', 'query', 'query']
    labels = meta['episode_positive_labels'].float()
    def forbidden(*args, **kwargs):
        raise AssertionError('Default branch must not read text or calibrated similarities')
    monkeypatch.setattr('trokens.models.pointformer.text_concat_episode', forbidden)
    monkeypatch.setattr(model, '_get_pot_label_text_features', forbidden)
    monkeypatch.setattr(model.blocks[0], 'forward_text_concat', forbidden)
    monkeypatch.setattr(model, '_compute_bidirectional_frame_similarity', forbidden)
    logits, patches, aux = unpack_episode_output(_forward(model, f, raw, meta))
    assert model.text_concat_st is None and not model.use_query_partial_q2s
    assert not model.blocks[0].checkpoint_mlp  # concat checkpoint options are ignored
    assert aux is None and model.query_evidence_verifier is None
    assert patches.shape == f.shape
    cls_oracle, patches_oracle = model.pt_forward(f, meta)
    torch.testing.assert_close(patches, patches_oracle)
    torch.testing.assert_close(logits, model.head(cls_oracle))
    changed = copy.deepcopy(meta)
    changed['episode_positive_labels'][2:] = ~changed['episode_positive_labels'][2:]
    changed['episode_class_ids'] = torch.tensor([999, 888])
    other_logits, other_patches, _ = _forward(model, f, raw, changed)
    torch.testing.assert_close(logits, other_logits, rtol=0, atol=0)
    torch.testing.assert_close(patches, other_patches, rtol=0, atol=0)
    split = support_query_split_multilabel(patches, labels, meta)
    score = train_score(model.cfg, split['support_preds'], split['query_preds'])
    torch.testing.assert_close(score, test_score(model.cfg, split['support_preds'], split['query_preds']), rtol=0, atol=0)
    q = F.relu(patches[2:]).mean(2).float()
    s = F.relu(torch.stack((patches[:2].mean(0), patches[1]))).mean(2).float()
    numerator = torch.einsum('qtd,ksd->qkts', q, s)
    denominator = q.norm(dim=-1)[:, None, :, None] * s.norm(dim=-1)[None, :, None, :] + .01
    distance = 1 - numerator / denominator
    oracle = -(distance.min(3).values.sum(2) + distance.min(2).values.sum(2))
    torch.testing.assert_close(score, oracle)
    assert score.le(0).all()
    model.cfg.FEW_SHOT.MATCHING.LOGIT_SCALE = 123.
    model.cfg.FEW_SHOT.MATCHING.LOGIT_BIAS = 456.
    model.pot_route_cfg.QUERY_PARTIAL_LOGIT_ALPHA = 789.
    model.pot_route_cfg.QUERY_PARTIAL_LOGIT_BIAS = 987.
    torch.testing.assert_close(score, train_score(model.cfg, split['support_preds'], split['query_preds']), rtol=0, atol=0)
    model.cfg.SOLVER.TEMPRATURE = .73
    loss, scaled, metrics = compute_query_partial_q2s_loss(score, split['query_batch_labels'], aux, model.cfg)
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(score / .73, labels[2:]))
    torch.testing.assert_close(scaled, score / .73)
    assert metrics['q2s_verified_loss'] == 0 and not any('mil' in k for k in metrics)
    (loss + F.binary_cross_entropy_with_logits(logits, labels)).backward()
    assert torch.isfinite(f.grad).all() and f.grad[:2].abs().sum() > 0 and f.grad[2:].abs().sum() > 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_default_branch_rejects_old_calibrated_shared_checkpoint():
    flags = (False, False, False, False)
    calibrated, baseline = _model(flags, archived_shared=True), _model(flags)
    assert calibrated.text_concat_st.architecture_signature.tolist() == [5, 0, 0, 1]
    assert baseline.text_concat_st is None
    validate_label_pre_temporal_checkpoint(baseline, baseline.state_dict())
    for destination, source in ((baseline, calibrated), (calibrated, baseline)):
        with pytest.raises(ValueError):
            validate_label_pre_temporal_checkpoint(destination, source.state_dict())


def test_default_constructor_skips_text_and_matches_explicit_original_initialization(monkeypatch):
    cfg = _cfg((False, False, False, False))
    original = cfg.clone()
    for name in original.FEW_SHOT.DECOUPLING:
        original.FEW_SHOT.DECOUPLING[name] = None
    # The old default entry used these explicit flags, not a text dispatcher.
    original.FEW_SHOT.TEXT_CONCAT_ST.ENABLE = False
    original.FEW_SHOT.POT_ROUTE.ENABLE = False
    original.FEW_SHOT.POT_ROUTE.QUERY_PARTIAL_ENABLE = False
    original.FEW_SHOT.QUERY_CLASS_MATCHABILITY.ENABLE = False
    original.FEW_SHOT.QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE = False
    original.FEW_SHOT.QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE = False
    def forbidden(*args, **kwargs):
        raise AssertionError('Default model constructor must not initialize any text resources')
    monkeypatch.setattr(Pointformer, '_load_dinotxt_visual_model', lambda self: nn.Identity())
    for name in ('_load_dinotxt_text_model', '_load_dinotxt_tokenizer',
                 '_load_atomic_label_names', '_load_label_prompt_groups'):
        monkeypatch.setattr(Pointformer, name, forbidden)
    torch.manual_seed(149)
    baseline = Pointformer(cfg)
    rng = torch.random.get_rng_state()
    torch.manual_seed(149)
    reference = Pointformer(original)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert not baseline.use_text_concat_st and not baseline.use_label_text_features
    assert not baseline.use_query_partial_q2s and not baseline.use_frame_softmax_route
    assert baseline.text_concat_st is None and baseline.query_evidence_verifier is None
    assert not hasattr(baseline, 'dinotxt_text_model')
    assert not hasattr(baseline, 'dinotxt_tokenizer')
    assert baseline.state_dict().keys() == reference.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


class _DDPEpisode(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.core = _model(flags)

    def forward(self, f, raw, meta):
        return _forward(self.core, f, raw, meta)


def _ddp_worker(rank, path, flags):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + path,
        rank=rank, world_size=2, timeout=timedelta(seconds=50))
    try:
        model = nn.parallel.DistributedDataParallel(_DDPEpisode(flags), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            f, raw, meta = _episode()
            out = model(f + .01 * rank, raw, meta)
            _loss(model.module.core, out, meta)[0].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():
            reference = p.detach().clone()
            torch.distributed.broadcast(reference, 0)
            torch.testing.assert_close(p, reference, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('flags', [(False, False, False, False), (True, False, True, True)])
def test_cpu_ddp_two_updates(tmp_path, flags):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / 'ddp'), flags), nprocs=2, join=True)
