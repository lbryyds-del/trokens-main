"""Synthetic oracles and real encoder integration; never load a GPU backbone."""
import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trokens.config.defaults import get_cfg, assert_and_infer_cfg
from trokens.config.ablation import resolve_ablation_cfg
from trokens.datasets.entity_tracks import build_entity_metadata, cluster_slots, distinct_slots
from trokens.models.entity_change_verifier import EntityChangeVerifier, initialize_from_rho_only
from trokens.models.text_concat_conditioning import text_concat_episode
from trokens.models.optimizer import construct_optimizer, set_lr
from trokens.utils.checkpoint import validate_text_concat_checkpoint
from test_ablation_switches import _model, _episode, _loss


CONFIG = Path(__file__).resolve().parents[1] / 'configs/trokens/sav_entity_change_s1.yaml'


def model(mode='none', **options):
    m = _model((True, True, True), mode)
    c = m.cfg.FEW_SHOT.ENTITY_CHANGE
    c.ENABLE = True
    c.MIN_POINTS = 2
    c.MAX_ENTITIES = 3
    c.GEO_MIN_STEPS = 1
    for k, v in options.items():
        setattr(c, k, v)
    if c.STRICT_COMMON_REPLAY or c.CORRESPONDENCE_SHUFFLE:
        m.cfg.TRAIN.ENABLE = False
    resolve_ablation_cfg(m.cfg)
    m._init_text_concat_st()
    return m


def episode(m):
    f, raw, meta = _episode()
    # Both classes have positive/negative references and visible moving tracks.
    meta['pred_visibility'].fill_(True)
    meta['point_indices'] = torch.tensor([[0, 0, 1, 2, 3]]).expand(4, -1)
    meta['obj_ids'] = torch.tensor([[7, 7, 7, 8, 8]]).expand(4, -1)
    g = torch.Generator().manual_seed(612)
    meta['pred_tracks'] = torch.randn(4, 3, 5, 2, generator=g) * .1
    meta['pred_tracks'][:, :, 1] = meta['pred_tracks'][:, :, 0]
    parts = [build_entity_metadata(meta['pred_tracks'][i], meta['pred_visibility'][i],
        meta['obj_ids'][i], meta['point_indices'][i], [0, 4, 8], m.cfg.FEW_SHOT.ENTITY_CHANGE)
        for i in range(4)]
    meta.update({k: torch.stack([p[k] for p in parts]) for k in parts[0]})
    return f, raw, meta


def test_distinct_count_does_not_confuse_duplicate_slots_with_tracks():
    ids = torch.tensor([10, 10, 10, 20, 20, 20, 20])
    tracks = torch.tensor([0, 0, 0, 1, 2, 3, 4])
    assert distinct_slots(tracks).tolist() == [True, False, False, True, True, True, True]
    assert cluster_slots(ids, tracks, 16, 4).tolist() == [-1, -1, -1, 0, 0, 0, 0]


def test_geometry_duplicate_invariance_zero_motion_and_time_spacing():
    c = get_cfg().FEW_SHOT.ENTITY_CHANGE
    c.GEO_MIN_STEPS = 1
    p = torch.zeros(8, 4, 2)
    p[:, :, 0] = torch.arange(8.)[:, None] * .01
    valid = torch.ones(8, 4).bool()
    a = build_entity_metadata(p, valid, torch.zeros(4).long(), torch.arange(4), torch.arange(8), c)
    assert a['entity_geo_valid'][0]
    assert torch.isfinite(a['entity_geo']).all()
    b = build_entity_metadata(p.repeat_interleave(2, 1), valid.repeat_interleave(2, 1),
        torch.zeros(8).long(), torch.arange(4).repeat_interleave(2), torch.arange(8), c)
    torch.testing.assert_close(a['entity_geo'], b['entity_geo'], rtol=0, atol=0)
    c2 = build_entity_metadata(p*2, valid, torch.zeros(4).long(), torch.arange(4), torch.arange(8)*2, c)
    torch.testing.assert_close(a['entity_geo'], c2['entity_geo'], rtol=0, atol=0)
    z = build_entity_metadata(p*0, valid, torch.zeros(4).long(), torch.arange(4), torch.arange(8), c)
    assert not z['entity_geo_valid'].any() and not z['entity_geo'].any()


@pytest.mark.parametrize('mode', ['none', 'full', 'selective'])
def test_zero_residual_exact_original_forward_and_auxiliary_gradients(mode):
    baseline = _model((True, True, True), mode).train()
    m = model(mode).train()
    f, raw, meta = episode(m)
    f.requires_grad_()
    old = text_concat_episode(baseline, f, meta, raw)
    out = text_concat_episode(m, f, meta, raw)
    for key in ('query_partial_q2s_logits', 'query_partial_q2s_base_logits',
                'query_relation_logits', 'global_classification_logits', 'text_concat_spatial_weights'):
        torch.testing.assert_close(out[key], old[key], rtol=0, atol=0)
    for channel in ('state', 'geo'):
        z = out['entity_' + channel + '_logits']
        assert z.requires_grad and out['entity_' + channel + '_valid'].all()
        # Crucially, the final-output diagnostic filter did NOT detach auxiliary logits.
        aux = F.binary_cross_entropy_with_logits(z, torch.eye(2))
        assert torch.autograd.grad(aux, f, allow_unused=True, retain_graph=True)[0] is None
    total, loss, metrics = _loss(m, out, meta)
    expected = _loss(baseline, old, meta)[1]
    for channel in ('state', 'geo'):
        expected = expected + .05 * F.binary_cross_entropy_with_logits(out['entity_' + channel + '_logits'], torch.eye(2))
    torch.testing.assert_close(loss, expected)
    total.backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().sum() > 0
    for name, parameter in m.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name, parameter in m.entity_change_verifier.named_parameters():
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize('missing', ['positive', 'negative', 'known', 'query', 'support', 'clusters', 'motion'])
def test_missing_evidence_neutral_and_not_supervised(missing):
    m = model('selective').train()
    f, raw, meta = episode(m)
    f.requires_grad_()
    if missing == 'positive': meta['episode_positive_labels'][:2].zero_()
    if missing == 'negative': meta['episode_positive_labels'][:2].fill_(True)
    if missing == 'known':
        meta['episode_label_known_mask'] = torch.ones(4, 2).bool()
        meta['episode_label_known_mask'][:2].zero_()
    if missing in ('query', 'support'):
        ix = slice(2, None) if missing == 'query' else slice(None, 2)
        meta['pred_visibility'][ix] = False
        meta['entity_geo_valid'][ix] = False
    if missing == 'clusters':
        meta['entity_index'].fill_(-1)
        meta['entity_geo_valid'].fill_(False)
    if missing == 'motion': meta['entity_geo_valid'].fill_(False)
    with torch.no_grad():
        m.entity_change_verifier.lambda_state.fill_(1.)
        m.entity_change_verifier.lambda_geo.fill_(1.)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = text_concat_episode(m, f, meta, raw)
        total, _, metrics = _loss(m, out, meta)
    for channel in ('state', 'geo'):
        if channel == 'state' and missing == 'motion': continue
        assert not out['entity_' + channel + '_valid'].any()
        assert not out['entity_' + channel + '_logits'].any()
        assert metrics['q2s_entity_' + channel + '_loss'] == 0
    total.backward()
    for name, parameter in m.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_frame_readout_oracle_common_only_and_constant_post_shift():
    m = model(GRANULARITY='frame', GEO_ENABLE=False).eval()
    f, _, meta = episode(m)
    p = f[:, None].expand(-1, 2, -1, -1, -1)
    w = torch.rand(p.shape[:-1], generator=torch.Generator().manual_seed(11))
    v = meta['pred_visibility'].clone()
    v[0, 0, 1] = False
    v[1, 2, 2] = False
    c = m.entity_change_verifier
    a = c.readout(p, w, v, meta['entity_index'], meta['entity_distinct'])
    common = v[:, 1:] & v[:, :-1] & v[:, :1]
    weights = (w[:, :, 1:] + w[:, :, :-1])*.5*common[:, None]
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-12)
    delta = torch.einsum('bktn,bktnd->bktd', weights, p[:, :, 1:]-p[:, :, :-1])
    unit = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    torch.testing.assert_close(a['state'].squeeze(-2), unit, rtol=0, atol=0)
    # Exact additive time-constant Post shifts, not nonlinear text interventions.
    shift = torch.randn(4, 2, 1, 5, 4, generator=torch.Generator().manual_seed(12))
    b = c.readout(p+shift, w, v, meta['entity_index'], meta['entity_distinct'])
    torch.testing.assert_close(a['state'], b['state'], atol=3e-7, rtol=3e-6)


def test_strict_replay_uses_original_metric_and_interval_average():
    m = model(GRANULARITY='frame', GEO_ENABLE=False, STRICT_COMMON_REPLAY=True).eval()
    f, _, meta = episode(m)
    c, rho = m.entity_change_verifier, m.query_evidence_verifier
    w = torch.rand(4, 2, 3, 5, generator=torch.Generator().manual_seed(1))
    read = c.readout(f[:, None].expand(-1, 2, -1, -1, -1), w,
                     meta['pred_visibility'], meta['entity_index'], meta['entity_distinct'])
    s, q = ({k:v[sl] for k,v in read.items()} for sl in (slice(0,2), slice(2,4)))
    labels = meta['episode_positive_labels'][:2].float()
    out = c.evidence(q, s, None, None, None, None, labels, torch.ones_like(labels).bool(), rho=rho)
    for k in range(2):
        qs, ss = q['state'][:,k], s['state'][:,k].flatten(0,2)[None]
        pos = (s['valid'][:,k] & labels[:,k,None,None].bool()).flatten()[None]
        neg = (s['valid'][:,k] & ~labels[:,k,None,None].bool()).flatten()[None]
        z,_,_ = rho._metric_relation_chunk(qs, torch.ones(2,1,2,1), ss, pos, neg)
        torch.testing.assert_close(out['entity_state_logits'][:,k], z[:,0].mean(-1), rtol=0,atol=0)


def test_query_truth_invariance_checkpoint_guards_and_explicit_import():
    m = model().eval()
    f, raw, meta = episode(m)
    modified = copy.deepcopy(meta)
    modified['episode_positive_labels'][2:] ^= True
    with torch.no_grad():
        a = text_concat_episode(m, f, meta, raw)
        b = text_concat_episode(m, f, modified, raw)
    for key in ('query_partial_q2s_logits', 'entity_state_logits', 'entity_geo_logits'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    old = _model((True, True, True), 'none').eval()
    validate_text_concat_checkpoint(m, m.state_dict())
    for a, b in ((m, old), (old, m)):
        with pytest.raises(ValueError, match='architecture'):
            validate_text_concat_checkpoint(a, b.state_dict())
    initialize_from_rho_only(m, old.state_dict())
    incomplete = dict(m.state_dict())
    incomplete.pop('entity_change_verifier.lambda_state')
    with pytest.raises(ValueError, match='Incomplete'):
        validate_text_concat_checkpoint(m, incomplete)


def test_optimizer_groups_update_resume_and_schedule_multiplier():
    m = model().train()
    opt = construct_optimizer(m, m.cfg)
    c = [g for g in opt.param_groups if g.get('group_name') == 'entity_change_scalars'][0]
    assert len(c['params']) == 6 and c['weight_decay'] == 0.
    for lr in (1e-6, 3e-5, 1e-6):
        set_lr(opt, {'lr': lr})
        assert c['lr'] == pytest.approx(100*lr)
        assert all(g['lr'] == lr for g in opt.param_groups if g is not c)
    f, raw, meta = episode(m)
    _loss(m, text_concat_episode(m, f, meta, raw), meta)[0].backward()
    before = m.entity_change_verifier.lambda_state.item()
    opt.step()
    assert m.entity_change_verifier.lambda_state.item() != before
    other = construct_optimizer(m, m.cfg)
    other.load_state_dict(opt.state_dict())
    set_lr(other, {'lr': 3e-5})
    assert other.param_groups[-1]['lr'] == pytest.approx(.003)


def test_state_lme_scalar_oracle_and_unknown_reference_exclusion():
    c = EntityChangeVerifier(get_cfg().FEW_SHOT.ENTITY_CHANGE)
    q = F.normalize(torch.tensor([[[[1., 1., 0.]]]]), dim=-1)
    refs = F.normalize(torch.tensor([[[[1., 0., 0.]], [[0., 1., 0.]]],
                                    [[[0., 0., 1.]], [[-1., 0., 0.]]],
                                    [[[1., 1., 0.]], [[1., 1., 0.]]]]), dim=-1)
    valid = torch.ones(3, 2, 1).bool()
    labels, known = torch.tensor([1., 0., -1.]), torch.tensor([True, True, True])
    out, present = c._metric(q, refs, valid, labels, known, 'state')
    sim = q[0,0,0] @ refs[:2].flatten(0,2).T
    expected = 10*.1*(torch.logsumexp(sim[:2]/.1,0)-torch.logsumexp(sim[2:]/.1,0))
    assert present
    torch.testing.assert_close(out.squeeze(), expected)


def test_inference_channel_knockout_and_shuffle_does_not_consume_rng():
    m = model().eval()
    f, raw, meta = episode(m)
    with torch.no_grad():
        m.entity_change_verifier.lambda_state.fill_(.5)
        m.entity_change_verifier.lambda_geo.fill_(.2)
        a = text_concat_episode(m, f, meta, raw)
        m.cfg.FEW_SHOT.ENTITY_CHANGE.GEO_TEST_MULT = 0.
        b = text_concat_episode(m, f, meta, raw)
        torch.testing.assert_close(a['query_partial_q2s_logits']-a['entity_geo_contribution'],
                                   b['query_partial_q2s_logits'])
        m.cfg.FEW_SHOT.ENTITY_CHANGE.STATE_TEST_MULT = 0.
        z = text_concat_episode(m, f, meta, raw)
        torch.testing.assert_close(z['query_partial_q2s_logits'], z['query_partial_q2s_rho_logits'], rtol=0, atol=0)
        m.cfg.FEW_SHOT.ENTITY_CHANGE.CORRESPONDENCE_SHUFFLE = True
        rng = torch.get_rng_state()
        a = text_concat_episode(m, f, meta, raw)
        b = text_concat_episode(m, f, meta, raw)
        assert torch.equal(rng, torch.get_rng_state())
        torch.testing.assert_close(a['entity_state_logits'], b['entity_state_logits'], rtol=0, atol=0)


@pytest.mark.parametrize('state,geo', [(True, False), (False, True)])
def test_single_channel_constructs_only_used_parameters(state, geo):
    m = model(STATE_ENABLE=state, GEO_ENABLE=geo).train()
    assert len(list(m.entity_change_verifier.parameters())) == 3
    f, raw, meta = episode(m)
    _loss(m, text_concat_episode(m, f, meta, raw), meta)[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.entity_change_verifier.parameters())


def test_empty_query_truth_mask_has_zero_aux_losses():
    m = model().train()
    f, raw, meta = episode(m)
    out = text_concat_episode(m, f, meta, raw)
    from tools.few_shot_multilabel import compute_query_partial_q2s_loss
    loss, _, metrics = compute_query_partial_q2s_loss(
        out['query_partial_q2s_logits'], torch.full((2,2), -1.), out, m.cfg)
    assert loss.item() == 0
    assert metrics['q2s_entity_state_loss'].item() == metrics['q2s_entity_geo_loss'].item() == 0
    loss.backward()
    assert all(p.grad is not None and not p.grad.any() for p in m.entity_change_verifier.parameters())


def test_config_protocol_and_optional_defaults():
    cfg = get_cfg(); cfg.merge_from_file(str(CONFIG)); cfg = assert_and_infer_cfg(cfg)
    assert (cfg.RNG_SEED,cfg.NUM_GPUS,cfg.FEW_SHOT.N_WAY,cfg.FEW_SHOT.K_SHOT)==(1,2,5,1)
    assert (cfg.SOLVER.BASE_LR,cfg.SOLVER.MAX_EPOCH,cfg.SOLVER.WARMUP_EPOCHS)==(3e-5,10,3)
    assert cfg.SOLVER.GRAD_ACCUM_STEPS == 2 and cfg.FEW_SHOT.TRAIN_EPISODES == 2000
    assert cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS == 6
    assert cfg.TEST.TEST_EPOCH_NUM == 10 and cfg.TRAIN.CHECKPOINT_KEEP_ALL
    assert Path(cfg.TRAIN.EVAL_DATA_CSV).is_file() and Path(cfg.TEST.DATA_CSV).is_file()
    assert not cfg.MODEL.FEATURE_SCALE.ENABLE and not cfg.MODEL.TRAIN_BACKBONE
    assert not cfg.TRAIN.AUTO_RESUME and not cfg.TRAIN.CHECKPOINT_FILE_PATH
    assert not get_cfg().FEW_SHOT.ENTITY_CHANGE.ENABLE
    clone = cfg.clone();cfg.freeze();resolve_ablation_cfg(cfg)
    assert cfg == clone and cfg.is_frozen()


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = model('selective')
    def forward(self, f, raw, meta):
        return text_concat_episode(self.net, f, meta, raw)


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://'+rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        wrapped = torch.nn.parallel.DistributedDataParallel(Wrapper(), find_unused_parameters=True)
        m = wrapped.module.net
        opt = construct_optimizer(wrapped, m.cfg)
        for step in range(2):
            f, raw, meta = episode(m)
            f = f + .01*rank
            if step == 1 and rank == 0:
                meta['entity_geo_valid'].zero_()
            opt.zero_grad(set_to_none=True)
            total = _loss(m, wrapped(f, raw, meta), meta)[0]
            total.backward(); opt.step()
        values = torch.stack([p.detach().float() for p in m.entity_change_verifier.parameters()])
        peers = [torch.empty_like(values) for _ in range(2)]
        torch.distributed.all_gather(peers, values)
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_two_process_ddp_updates(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path/'ddp'),), nprocs=2, join=True)
