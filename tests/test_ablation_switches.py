"""Compact maintained regression for the public SAV ablation switches."""
import copy
import itertools
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
from tools.few_shot_multilabel import compute_query_partial_q2s_loss, compute_learned_evidence_mil_loss


CONFIG = Path(__file__).resolve().parents[1] / 'configs/trokens/sav.yaml'
COMBINATIONS = list(itertools.product((False, True), repeat=4))  # Query, Support, m, rho
VALID = [x for x in COMBINATIONS if x[0] or not any(x[2:])]


def _cfg(flags):
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG))
    for name, value in zip(('QUERY_ENABLE', 'SUPPORT_ENABLE', 'M_ENABLE', 'RHO_ENABLE'), flags):
        cfg.FEW_SHOT.DECOUPLING[name] = value
    return resolve_ablation_cfg(cfg)


def _model(flags, mode='selective'):
    """Real encoder/verifier/head, with small tensors and no backbone downloads."""
    torch.manual_seed(71)
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = _cfg(flags)
    cfg = model.cfg
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
    model.use_query_partial_q2s = model.use_text_concat_st = model.is_multilabel_few_shot = True
    model.feat_extractor_type = 'dinotxt_vitl14_reg4'
    model.cls_token = nn.Parameter(torch.randn(1, 1, 4))
    model.pos_drop, model.head_drop = nn.Identity(), nn.Identity()
    model.blocks = nn.ModuleList([TrajectoryAttentionBlock(
        cfg=cfg, dim=4, num_heads=2, mlp_ratio=2., qkv_bias=True, use_pt_visibility=True)])
    model.norm, model.head, model.pre_logits = nn.LayerNorm(4), nn.Linear(4, 2), nn.Linear(4, 4)
    model._get_pot_label_text_features = lambda ids, dtype: torch.eye(4, dtype=dtype, device=ids.device)[ids]
    model._init_query_evidence_verifier()
    model._init_text_concat_st()
    return model


def _episode():
    rng = torch.Generator().manual_seed(414)
    f, raw = [torch.randn(4, 3, 5, 4, generator=rng) for _ in range(2)]
    mask = torch.ones(4, 3, 5).bool()
    visibility = mask.clone()
    visibility[0, 1, 2:] = False
    visibility[2, 1] = False
    return f, raw, dict(support_mask=torch.tensor([True, True, False, False]),
        pred_query_mask=mask, pred_visibility=visibility, episode_class_ids=torch.tensor([0, 1]),
        episode_positive_labels=torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]]).bool())


def _loss(model, out, meta):
    targets = meta['episode_positive_labels'][~meta['support_mask']].float()
    q2s, _, metrics = compute_query_partial_q2s_loss(out['query_partial_q2s_logits'], targets, out, model.cfg)
    total = q2s + F.binary_cross_entropy_with_logits(out['global_classification_logits'],
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
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.SHARED_QUERY_ENABLE == (not query)
    assert cfg.FEW_SHOT.TEXT_CONCAT_ST.SHARED_SUPPORT_ENABLE == (not support)
    assert (match.QUALITY_ENABLE, match.UNIQUENESS_ENABLE) == (m, rho)
    assert (match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT) == ((.6, .4) if m or rho else (1., 0.))
    assert match.BASE_ONLY_LOSS_ENABLE == (not (m or rho))
    assert match.DUAL_LOGIT_LOSS_ENABLE == (m or rho)
    assert match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT == (.05 if m else 0.)
    assert match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT == (.05 if rho else 0.)
    assert match.ABSOLUTE_MASS_TRANSPORT_STRENGTH == (1. if m or rho else 0.)


@pytest.mark.parametrize('flags', VALID)
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


@pytest.mark.parametrize('flags', VALID)
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


@pytest.mark.parametrize('flags', VALID)
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
        pred_query_mask=torch.ones(7, 3, 5).bool(), pred_visibility=torch.ones(7, 3, 5).bool(),
        episode_class_ids=torch.arange(5), episode_positive_labels=torch.eye(5)[[0, 1, 2, 3, 4, 0, 3]].bool())
    out = text_concat_episode(model, f, meta, raw)
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


class _DDPEpisode(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.core = _model(flags)

    def forward(self, f, raw, meta):
        return text_concat_episode(self.core, f, meta, raw)


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
