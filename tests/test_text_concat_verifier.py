"""Concat v2: unchanged encoder/readout, Raw m, same-class Post rho, losses."""
import copy
from datetime import timedelta

import pytest
import torch
from torch import nn

from test_text_concat_conditioning import _concat_model, _loss
from test_learned_query_verifier import _episode
from trokens.models.text_concat_conditioning import text_concat_episode, _encode_chunk
from trokens.models.query_matching import normalize_patch_region_weights
from trokens.models.query_evidence_verifier import set_verifier_training_progress
from trokens.utils.checkpoint import load_checkpoint, validate_label_pre_temporal_checkpoint
from tools.few_shot_multilabel import compute_query_partial_q2s_loss, compute_learned_evidence_mil_loss


def _verified_model(quality=True, uniqueness=True, mode='none'):
    model = _concat_model(checkpoint=mode == 'full', mlp_checkpoint=mode == 'selective',
                          readout_checkpoint=mode == 'selective')
    cfg = model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY
    cfg.QUALITY_ENABLE, cfg.UNIQUENESS_ENABLE = quality, uniqueness
    cfg.BASE_ONLY_LOSS_ENABLE = False
    cfg.DUAL_LOGIT_LOSS_ENABLE = True
    cfg.BASE_LOGIT_LOSS_WEIGHT, cfg.VERIFIED_LOGIT_LOSS_WEIGHT = .6, .4
    cfg.ABSOLUTE_MASS_TRANSPORT_STRENGTH = 1.
    cfg.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT = .05
    cfg.LEARNED_VERIFIER.RELATION_MIL_WEIGHT = .05
    cfg.LEARNED_VERIFIER.QUALITY_MODE = 'relation_decision'
    cfg.LEARNED_VERIFIER.ACTIVATION_CHECKPOINT = mode == 'full'
    model._init_query_evidence_verifier()
    model._init_text_concat_st()
    return model


@pytest.mark.parametrize('quality,uniqueness', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('mode', ['none', 'full', 'selective'])
def test_judgments_and_all_required_gradients(quality, uniqueness, mode, monkeypatch):
    model = _verified_model(quality, uniqueness, mode).train()
    def forbidden(*args, **kwargs):
        raise AssertionError('Old routing or Support/text fusion must not run')
    monkeypatch.setattr(model, '_fuse_episode_text_with_support_visual', forbidden)
    monkeypatch.setattr(model, '_compute_batched_frame_softmax_text_prototypes', forbidden)
    f, raw, meta = _episode()
    f.requires_grad_()
    out = text_concat_episode(model, f, meta, raw)
    assert out['query_evidence_enabled']
    assert ('query_quality_logits' in out) == quality
    assert ('query_relation_logits' in out) == uniqueness
    assert all(not torch.is_tensor(v) or v.ndim < 5 for v in out.values())
    m = out.get('query_frame_absolute_patch_mass', 1.)
    rho = out.get('query_frame_matchability', 1.)
    torch.testing.assert_close(out['query_frame_target_mass'], m * rho)
    assert (out['query_partial_q2s_logits'] <= out['query_partial_q2s_base_logits'] + 1e-6).all()
    _loss(model, out, meta).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert f.grad[:2].abs().sum() > 0 and f.grad[2:].abs().sum() > 0


def test_base_features_and_spatial_readout_unchanged_with_no_extra_encoder_pass():
    off, on = _concat_model().eval(), _verified_model().eval()
    f, raw, meta = _episode()
    count = []
    hook = on.blocks[0].temporal_attn.register_forward_hook(lambda *args: count.append(1))
    on.text_concat_st.class_chunk_size = 2
    on.text_concat_st.video_chunk_size = 8
    a, b = text_concat_episode(off, f, meta), text_concat_episode(on, f, meta, raw)
    hook.remove()
    assert len(count) == 2  # Support chunk + Query chunk; two classes batched.
    for key in ('query_partial_q2s_base_logits', 'query_partial_query_prototypes',
                'query_partial_support_prototypes', 'global_classification_logits',
                'text_concat_spatial_weights'):
        torch.testing.assert_close(a[key], b[key], msg=key)
    assert not torch.allclose(b['query_partial_q2s_logits'], b['query_partial_q2s_base_logits'])


def test_raw_quality_is_independent_of_post_and_rho_independent_of_raw():
    model = _verified_model().eval()
    f, raw, meta = _episode()
    a = text_concat_episode(model, f, meta, raw)
    b = text_concat_episode(model, f + torch.randn_like(f), meta, raw)
    c = text_concat_episode(model, f, meta, raw.flip(-1))
    torch.testing.assert_close(a['query_quality_logits'], b['query_quality_logits'], atol=0, rtol=0)
    torch.testing.assert_close(a['query_relation_logits'], c['query_relation_logits'], atol=0, rtol=0)
    assert not torch.allclose(a['query_relation_logits'], b['query_relation_logits'])
    with pytest.raises(ValueError, match='Raw'):
        text_concat_episode(model, f, meta)


def test_query_truth_isolation_and_candidate_video_permutations():
    model = _verified_model().eval()
    f, raw, meta = _episode()
    a = text_concat_episode(model, f, meta, raw)
    changed = copy.deepcopy(meta)
    changed['episode_positive_labels'][2:] = ~changed['episode_positive_labels'][2:]
    changed['episode_label_known_mask'] = torch.ones(4, 2).bool()
    changed['episode_label_known_mask'][2:] = False
    b = text_concat_episode(model, f, changed, raw)
    for key in a:
        if torch.is_tensor(a[key]):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)
    changed = copy.deepcopy(meta)
    changed['episode_class_ids'] = meta['episode_class_ids'].flip(0)
    changed['episode_positive_labels'] = meta['episode_positive_labels'].flip(1)
    c = text_concat_episode(model, f, changed, raw)
    for key in ('query_partial_q2s_logits', 'query_quality_logits', 'query_relation_logits'):
        torch.testing.assert_close(a[key].flip(1), c[key])
    order = torch.tensor([3, 1, 2, 0])
    changed = {k: v[order] if v.shape[0] == 4 else v for k, v in meta.items()}
    d = text_concat_episode(model, f[order], changed, raw[order])
    for key in ('query_partial_q2s_logits', 'query_quality_logits', 'query_relation_logits'):
        torch.testing.assert_close(a[key].flip(0), d[key])
    torch.testing.assert_close(a['global_classification_logits'][order], d['global_classification_logits'])


def test_visibility_is_evidence_only_and_missing_signs_are_neutral():
    model = _verified_model().eval()
    f, raw, meta = _episode()
    a = text_concat_episode(model, f, meta, raw)
    meta['pred_visibility'][2, 1] = False
    b = text_concat_episode(model, f, meta, raw)
    torch.testing.assert_close(a['query_partial_q2s_base_logits'], b['query_partial_q2s_base_logits'])
    torch.testing.assert_close(a['text_concat_spatial_weights'], b['text_concat_spatial_weights'])
    assert b['query_frame_absolute_patch_mass'][0, :, 1].eq(0).all()
    assert b['query_evidence_patch_weights'][0, :, 1].eq(0).all()
    meta['episode_label_known_mask'] = torch.ones(4, 2).bool()
    meta['episode_label_known_mask'][1, 0] = False  # only known negative for c=0
    c = text_concat_episode(model, f, meta, raw)
    assert not c['query_relation_valid'][:, 0].any()
    assert not c['query_quality_valid'][:, 0].any()
    assert c['query_frame_matchability'][:, 0].eq(1).all()
    assert c['query_frame_absolute_patch_mass'][:, 0, 0].eq(1).all()


def test_same_class_reference_and_patch_metric_independent_oracle():
    model = _verified_model().eval()
    f, raw, meta = _episode()
    meta['pred_visibility'][0, 1, 1:] = False
    text = model._get_pot_label_text_features(meta['episode_class_ids'], f.dtype)
    _, weights, _, post = _encode_chunk(model, f, meta['pred_query_mask'], text,
                                      evidence_readout=lambda patches, weights: patches)
    pi = normalize_patch_region_weights(weights, meta['pred_visibility'] & meta['pred_query_mask'])
    references = torch.nn.functional.normalize((pi[:2, ..., None] * post[:2]).sum(-2), dim=-1)
    query = torch.nn.functional.normalize(post[2:], dim=-1)
    tau = torch.nn.functional.softplus(model.query_evidence_verifier.metric_temperature_raw) + 1e-4
    scale = torch.nn.functional.softplus(model.query_evidence_verifier.metric_scale_raw) + 1e-4
    expected = torch.empty(2, 2, 3)
    for q in range(2):
        for k in range(2):
            for t in range(3):
                scores = query[q, k, t] @ references[:, k].flatten(0, 1).T
                scores = scores.reshape(5, 2, 3)
                positive = meta['episode_positive_labels'][:2, k]
                a = tau * (torch.logsumexp(scores[:, positive].flatten(1) / tau, -1) - torch.tensor(3.).log())
                b = tau * (torch.logsumexp(scores[:, ~positive].flatten(1) / tau, -1) - torch.tensor(3.).log())
                expected[q, k, t] = scale * (pi[q+2, k, t] * (a-b)).sum()
    out = text_concat_episode(model, f, meta, raw)
    torch.testing.assert_close(out['query_relation_logits'], expected, atol=3e-6, rtol=3e-5)


def test_dual_loss_formula_detached_auxiliary_and_transport_warmup():
    model = _verified_model().train()
    f, raw, meta = _episode()
    f.requires_grad_(); raw.requires_grad_()
    set_verifier_training_progress(model, 0, 100)
    out = text_concat_episode(model, f, meta, raw)
    torch.testing.assert_close(out['query_partial_q2s_logits'], out['query_partial_q2s_base_logits'], atol=0, rtol=0)
    labels = meta['episode_positive_labels'][2:].float()
    m_loss = compute_learned_evidence_mil_loss(out['query_quality_logits'], out['query_quality_valid'], labels)['loss']
    r_loss = compute_learned_evidence_mil_loss(out['query_relation_logits'], out['query_relation_valid'], labels)['loss']
    actual, _, diag = compute_query_partial_q2s_loss(out['query_partial_q2s_logits'], labels, out, model.cfg)
    bce = nn.functional.binary_cross_entropy_with_logits
    expected = (.6 * bce(out['query_partial_q2s_base_logits'], labels)
                + .4 * bce(out['query_partial_q2s_logits'], labels) + .05 * (m_loss + r_loss))
    torch.testing.assert_close(actual, expected)
    (m_loss + r_loss).backward()
    assert f.grad is None and raw.grad is None  # stop before evidence features, not head parameters
    assert model.query_evidence_verifier.semantic_quality.projection.weight.grad.abs().sum() > 0
    assert model.query_evidence_verifier.metric_scale_raw.grad.abs() > 0


@pytest.mark.parametrize('mode', ['none', 'full', 'selective'])
def test_empty_visual_or_evidence_masks_have_finite_gradients(mode):
    for visual in (False, True):
        model = _verified_model(mode=mode).train()
        f, raw, meta = _episode()
        f.requires_grad_()
        meta['pred_visibility'].zero_()
        if visual:
            meta['pred_query_mask'].zero_()
        out = text_concat_episode(model, f, meta, raw)
        assert out['query_frame_target_mass'].eq(0).all()
        assert not out['query_quality_valid'].any() and not out['query_relation_valid'].any()
        _loss(model, out, meta).backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert torch.isfinite(f.grad).all()


def test_forward_passes_pre_position_raw_and_keeps_motion(monkeypatch):
    model = _verified_model().eval()
    f, _, meta = _episode()
    model.cfg.MF.USE_BASE_POS_EMBED = False
    model.cfg.MF.USE_PT_SPACE_POS_EMBED = True
    meta['pred_tracks'] = torch.randn(4, 3, 5, 2)
    meta['hod_feat'] = torch.randn_like(f)
    calls = []
    def sample(value, tracks, add_pt_pos_embed):
        calls.append(add_pt_pos_embed)
        return value + .11 if add_pt_pos_embed else value
    monkeypatch.setattr(model, 'get_dinotxt_features', lambda x: f)
    monkeypatch.setattr(model, '_sample_point_features', sample)
    model.hod_motion_module = nn.Identity()
    class Motion(nn.Module):
        def forward(self, tracks, visibility):
            return torch.ones_like(f) * .2
    model.cross_motion_module = Motion()
    actual = model({'video': f, 'metadata': meta})
    expected = text_concat_episode(model, f + .11 + meta['hod_feat'] + .2, meta, f)
    assert calls == [False, True]
    for key in ('query_partial_q2s_logits', 'query_quality_logits', 'query_relation_logits'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cuda_amp_full_strength_verifier_checkpoint_equivalence(dtype):
    models = [_verified_model(mode=mode).cuda().train() for mode in ('none', 'selective')]
    f, raw, meta = _episode('cuda')
    outputs = []
    for model in models:
        model.query_evidence_verifier.transport_progress = 1.
        with torch.autocast('cuda', dtype=dtype):
            out = text_concat_episode(model, f, meta, raw)
            loss = _loss(model, out, meta)
        loss.backward()
        outputs.append(out)
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for key in ('query_partial_q2s_logits', 'query_quality_logits', 'query_relation_logits'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    for (name, p), (_, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


@pytest.mark.parametrize('mode', ['full', 'selective'])
def test_checkpoint_equivalence_with_stochastic_encoder(mode):
    from trokens.models.common import DropPath
    plain, checked = _verified_model(), _verified_model(mode=mode)
    for model in (plain, checked):
        model.train()
        model.blocks[0].drop_path = DropPath(.2)
    f, raw, meta = _episode()
    outputs = []
    for model in (plain, checked):
        torch.manual_seed(123)
        out = text_concat_episode(model, f, meta, raw)
        _loss(model, out, meta).backward()
        outputs.append((out, torch.get_rng_state()))
    assert torch.equal(outputs[0][1], outputs[1][1])
    torch.testing.assert_close(outputs[0][0]['query_partial_q2s_logits'], outputs[1][0]['query_partial_q2s_logits'], atol=0, rtol=0)
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=4e-6, rtol=4e-5, msg=name)


def test_serial_class_and_video_chunks_match_batched_verifier():
    serial, batched = _verified_model(mode='selective'), _verified_model(mode='selective')
    serial.text_concat_st.class_chunk_size = serial.text_concat_st.video_chunk_size = 1
    batched.text_concat_st.class_chunk_size = 2
    batched.text_concat_st.video_chunk_size = 8
    f, raw, meta = _episode()
    a, b = (text_concat_episode(model, f, meta, raw) for model in (serial, batched))
    for key in ('query_partial_q2s_logits', 'query_quality_logits', 'query_relation_logits',
                'text_concat_spatial_weights', 'global_classification_logits'):
        torch.testing.assert_close(a[key], b[key], atol=4e-6, rtol=4e-5, msg=key)
    _loss(serial, a, meta).backward()
    _loss(batched, b, meta).backward()
    for (name, p), (_, q) in zip(serial.named_parameters(), batched.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=5e-5, msg=name)


def test_checkpoint_v1_initialization_requires_reset_and_trained_heads_for_resume(tmp_path):
    base, model = _concat_model(), _verified_model()
    with pytest.raises(ValueError, match='architecture'):
        validate_label_pre_temporal_checkpoint(model, base.state_dict())
    path = tmp_path / 'base.pyth'
    torch.save(dict(model_state=base.state_dict(), epoch=10), path)
    epoch = load_checkpoint(str(path), model, data_parallel=False, epoch_reset=True,
                            allow_label_pre_temporal_init=True, allow_verifier_init=True)
    assert epoch == -1 and not model._verifier_checkpoint_loaded
    assert model.text_concat_st.architecture_signature.tolist() == [2, 1, 1, 1, 1, 1]
    bad = model.state_dict()
    bad.pop('query_evidence_verifier.metric_scale_raw')
    with pytest.raises(ValueError, match='Incomplete'):
        validate_label_pre_temporal_checkpoint(model, bad)
    torch.save(dict(model_state=model.state_dict(), epoch=2), path)
    restored = _verified_model()
    assert load_checkpoint(str(path), restored, data_parallel=False) == 2
    f, raw, meta = _episode()
    torch.testing.assert_close(text_concat_episode(model, f, meta, raw)['query_partial_q2s_logits'],
                               text_concat_episode(restored, f, meta, raw)['query_partial_q2s_logits'])


class _Episode(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _verified_model(mode='selective')
    def forward(self, f, raw, meta):
        return text_concat_episode(self.encoder, f, meta, raw)


def _ddp_worker(rank, path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + path, rank=rank,
                                        world_size=2, timeout=timedelta(seconds=50))
    try:
        model = torch.nn.parallel.DistributedDataParallel(_Episode(), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            f, raw, meta = _episode()
            out = model(f + .01 * rank, raw, meta)
            _loss(model.module.encoder, out, meta).backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for p in model.parameters():
                other = p.detach().clone()
                torch.distributed.broadcast(other, 0)
                assert torch.equal(p, other)
    finally:
        torch.distributed.destroy_process_group()


def test_two_process_ddp_with_judgments_and_selective_checkpoint(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / 'init'),), nprocs=2, join=True)
