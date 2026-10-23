"""Text concat v1: independent readout oracle, role symmetry and loss contract."""
import copy
from datetime import timedelta

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_label_pre_temporal import _new_model
from test_learned_query_verifier import _episode
from trokens.models.attention import Attention
from trokens.models.text_concat_conditioning import _encode_chunk, text_concat_episode
from trokens.utils.checkpoint import validate_label_pre_temporal_checkpoint, load_checkpoint
from tools.few_shot_multilabel import (
    compute_query_partial_q2s_loss, multilabel_classification_loss,
    single_path_episode_targets, unpack_episode_output,
)


def _concat_model(checkpoint=False, cls=True, classification=True, spatial_tokens=1,
                  mlp_checkpoint=False, readout_checkpoint=False):
    model = _new_model(False, False, cls=cls, classification=classification)
    model.use_label_pre_temporal = False
    model.label_pre_temporal = None
    model.use_frame_softmax_route = model.use_support_text_fusion = False
    model.use_text_concat_st = model.is_multilabel_few_shot = True
    model.feat_extractor_type = 'dinotxt_vitl14_reg4'
    cfg = model.cfg.FEW_SHOT
    cfg.LABEL_PRE_TEMPORAL.ENABLE = cfg.POT_ROUTE.ENABLE = cfg.SUPPORT_TEXT_FUSION.ENABLE = False
    cfg.TEXT_CONCAT_ST.ENABLE = True
    cfg.TEXT_CONCAT_ST.SPATIAL_TEXT_TOKENS = spatial_tokens
    cfg.TEXT_CONCAT_ST.ACTIVATION_CHECKPOINT = checkpoint
    cfg.TEXT_CONCAT_ST.MLP_CHECKPOINT = mlp_checkpoint
    cfg.TEXT_CONCAT_ST.READOUT_CHECKPOINT = readout_checkpoint
    match = cfg.QUERY_CLASS_MATCHABILITY
    match.BASE_ONLY_LOSS_ENABLE = True
    match.DUAL_LOGIT_LOSS_ENABLE = False
    match.BASE_LOGIT_LOSS_WEIGHT, match.VERIFIED_LOGIT_LOSS_WEIGHT = 1., 0.
    match.ABSOLUTE_MASS_TRANSPORT_STRENGTH = 0.
    match.LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT = 0.
    match.LEARNED_VERIFIER.RELATION_MIL_WEIGHT = 0.
    model._init_text_concat_st()
    return model


def _loss(model, out, meta):
    targets = meta['episode_positive_labels'][~meta['support_mask']].float()
    loss = compute_query_partial_q2s_loss(out['query_partial_q2s_logits'], targets, out, model.cfg)[0]
    if model.text_concat_st.global_classifier:
        loss = loss + multilabel_classification_loss(out['global_classification_logits'],
            meta['episode_positive_labels'].float(), model.cfg, nn.BCEWithLogitsLoss())
    return loss


@pytest.mark.parametrize('use_mask', [False, True])
def test_readout_is_pre_dropout_visual_renormalized_text_query_attention(use_mask):
    torch.manual_seed(9)
    attn = Attention(4, num_heads=2, qkv_bias=True, attn_drop=.7, use_mask=use_mask).train()
    x = torch.randn(2, 9, 4, requires_grad=True)  # CLS + 5 visual + 3 text
    visual = torch.tensor([[True, False, True, True, False], [False] * 5])
    mask = torch.cat((torch.ones(2, 1).bool(), visual, torch.ones(2, 3).bool()), 1)
    _, actual = attn.forward_with_text_readout(x, mask, visual, 1, 3)
    qkv = attn.qkv(x).reshape(2, 9, 3, 2, 2).permute(2, 0, 3, 1, 4)
    logits = (qkv[0] @ qkv[1].transpose(-2, -1)) * attn.scale
    if use_mask:
        logits = logits.masked_fill(~mask[:, None, None], float('-inf'))
    full = logits.softmax(-1)[:, :, -3:, 1:6] * visual[:, None, None]
    expected = (full / full.sum(-1, keepdim=True).clamp_min(1e-30)).mean((1, 2))
    torch.testing.assert_close(actual, expected)
    assert actual[1].eq(0).all() and actual[0, ~visual[0]].eq(0).all()
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    _, repeat = attn.forward_with_text_readout(x, mask, visual, 1, 3)
    torch.testing.assert_close(actual, repeat, atol=0, rtol=0)  # dropout-free readout


@pytest.mark.parametrize('cls', [False, True])
@pytest.mark.parametrize('spatial_tokens', [1, 32])
def test_encoder_and_post_pooling_match_independent_serial_oracle(cls, spatial_tokens):
    model = _concat_model(cls=cls, spatial_tokens=spatial_tokens).eval()
    features, _, meta = _episode()
    mask = meta['pred_query_mask'].clone()
    mask[0, 1] = False
    text = model._get_pot_label_text_features(torch.tensor([0, 1]), features.dtype)
    actual, weights, global_sum = _encode_chunk(model, features, mask, text)
    expected_frames, expected_weights, expected_global = [], [], []
    block = model.blocks[0]
    for k in range(2):
        f, r, g = [], [], []
        for b in range(4):
            original = features[b].transpose(0, 1)  # N,T,D
            temporal = torch.cat((original, text[k].expand(5, 1, 4)), 1)
            tm = torch.cat((mask[b].T, torch.ones(5, 1).bool()), 1)
            temporal_res = block.temporal_attn(block.temporal_norm1(temporal), tm)[:, :3]
            post_temporal = original + block.temporal_fc(temporal_res)
            spatial = post_temporal.transpose(0, 1)
            sm = mask[b]
            if cls:
                spatial = torch.cat((model.cls_token.expand(3, 1, 4), spatial), 1)
                sm = torch.cat((torch.ones(3, 1).bool(), sm), 1)
            spatial = torch.cat((spatial, text[k].expand(3, spatial_tokens, 4)), 1)
            sm = torch.cat((sm, torch.ones(3, spatial_tokens).bool()), 1)
            norm = block.norm1(spatial)
            spatial_res = block.attn(norm, sm)
            # Independent readout from explicit projected Q,K and visual softmax.
            qkv = block.attn.qkv(norm).reshape(3, 5 + int(cls) + spatial_tokens, 3, 2, 2).permute(2, 0, 3, 1, 4)
            score = (qkv[0][:, :, -spatial_tokens:] @ qkv[1][:, :, int(cls):int(cls)+5].transpose(-2, -1)) * block.attn.scale
            score = score.masked_fill(~mask[b][:, None, None], float('-inf'))
            score = torch.where(mask[b].any(-1)[:, None, None, None], score, torch.zeros_like(score))
            weight = score.softmax(-1).masked_fill(~mask[b][:, None, None], 0.).mean((1, 2))
            patches = (post_temporal + spatial_res[:, int(cls):int(cls)+5].transpose(0, 1)).reshape(1, 15, 4)
            if cls:
                patches = torch.cat((model.cls_token + spatial_res[:, :1].mean(0, keepdim=True), patches), 1)
            encoded = model.norm(patches + block.mlp(block.norm2(patches)))
            g.append(encoded[:, 0] if cls else encoded.mean(1))
            values = encoded[:, int(cls):].reshape(5, 3, 4).transpose(0, 1)
            f.append((weight[..., None] * values).sum(1))
            r.append(weight)
        expected_frames.append(torch.stack(f))
        expected_weights.append(torch.stack(r))
        expected_global.append(torch.cat(g))
    torch.testing.assert_close(actual, torch.stack(expected_frames, 1))
    torch.testing.assert_close(weights, torch.stack(expected_weights, 1))
    torch.testing.assert_close(global_sum, torch.stack(expected_global, 1).sum(1))


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('classification', [False, True])
def test_episode_gradients_loss_and_no_verifier_or_pre_router(checkpoint, classification, monkeypatch):
    model = _concat_model(checkpoint, classification=classification).train()
    def forbidden(*args, **kwargs):
        raise AssertionError('Old routing/conditioning/transport must not run')
    monkeypatch.setattr(model, '_compute_batched_frame_softmax_text_prototypes', forbidden)
    monkeypatch.setattr(model, '_fuse_episode_text_with_support_visual', forbidden)
    features, _, meta = _episode()
    features.requires_grad_()
    meta['sample_type'] = ['support', 'support', 'query', 'query']
    out = text_concat_episode(model, features, meta)
    assert model.query_evidence_verifier is None and model.label_pre_temporal is None
    assert out['query_partial_q2s_logits'] is out['query_partial_q2s_base_logits']
    assert not any('quality' in key or 'relation' in key for key in out)
    assert not out['query_evidence_enabled']
    assert out['query_partial_query_prototypes'].shape == (2, 2, 3, 4)
    assert out['query_partial_support_prototypes'].shape == (2, 3, 4)
    torch.testing.assert_close(out['text_concat_spatial_weights'].sum(-1), torch.ones(4, 2, 3))
    logits, _, unpacked = unpack_episode_output(out)
    targets = single_path_episode_targets(out, meta['episode_positive_labels'].float(), meta)['query_batch_labels']
    assert targets.shape == (2, 2) and unpacked is out
    qloss, _, metrics = compute_query_partial_q2s_loss(out['query_partial_q2s_logits'], targets, out, model.cfg)
    torch.testing.assert_close(qloss, F.binary_cross_entropy_with_logits(out['query_partial_q2s_logits'], targets))
    assert not any('mil' in key for key in metrics)
    if classification:
        assert logits.shape == (4, 2)
    _loss(model, out, meta).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert features.grad[:2].abs().sum() > 0 and features.grad[2:].abs().sum() > 0


def test_no_query_truth_leakage_class_and_video_permutations():
    model = _concat_model().eval()
    f, _, meta = _episode()
    a = text_concat_episode(model, f, meta)
    changed = copy.deepcopy(meta)
    changed['episode_positive_labels'][2:] = ~changed['episode_positive_labels'][2:]
    changed['episode_label_known_mask'] = torch.ones(4, 2).bool()
    changed['episode_label_known_mask'][2:] = False
    b = text_concat_episode(model, f, changed)
    for key in ('query_partial_q2s_logits', 'text_concat_spatial_weights', 'global_classification_logits'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    perm = copy.deepcopy(meta)
    perm['episode_class_ids'] = meta['episode_class_ids'].flip(0)
    perm['episode_positive_labels'] = meta['episode_positive_labels'].flip(1)
    c = text_concat_episode(model, f, perm)
    torch.testing.assert_close(a['query_partial_q2s_logits'].flip(1), c['query_partial_q2s_logits'])
    order = torch.tensor([3, 1, 2, 0])
    perm = {k: v[order] if v.shape[0] == 4 else v for k, v in meta.items()}
    d = text_concat_episode(model, f[order], perm)
    torch.testing.assert_close(a['query_partial_q2s_logits'].flip(0), d['query_partial_q2s_logits'])
    torch.testing.assert_close(a['text_concat_spatial_weights'][order], d['text_concat_spatial_weights'])


def test_roles_use_identical_encoding_and_empty_visual_rows_are_zero():
    model = _concat_model().eval()
    f, _, meta = _episode()
    f[2] = f[0]
    meta['pred_query_mask'][0, 1] = meta['pred_query_mask'][2, 1] = False
    out = text_concat_episode(model, f, meta)
    torch.testing.assert_close(out['text_concat_spatial_weights'][0], out['text_concat_spatial_weights'][2])
    torch.testing.assert_close(out['query_partial_support_prototypes'][0], out['query_partial_query_prototypes'][0, 0])
    assert out['query_partial_query_prototypes'][0, :, 1].eq(0).all()


def test_all_visual_masks_empty_stays_finite_through_backward():
    model = _concat_model(True).train()
    f, _, meta = _episode()
    f.requires_grad_()
    meta['pred_query_mask'].zero_()
    out = text_concat_episode(model, f, meta)
    assert out['text_concat_spatial_weights'].eq(0).all()
    assert out['query_partial_query_prototypes'].eq(0).all()
    torch.testing.assert_close(out['query_partial_q2s_logits'], torch.full((2, 2), -2.))
    _loss(model, out, meta).backward()
    assert torch.isfinite(f.grad).all()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_identical_text_copies_do_not_create_distinct_visual_readout_views():
    torch.manual_seed(21)
    attention = Attention(8, num_heads=2).eval()
    visual, text = torch.randn(2, 6, 8), torch.randn(2, 1, 8)
    mask = torch.ones(2, 6).bool()
    weights, values = [], []
    for count in (1, 32):
        tokens = torch.cat((visual, text.expand(2, count, 8)), 1)
        value, weight = attention.forward_with_text_readout(tokens, None, mask, 0, count)
        weights.append(weight)
        values.append(value[:, :6])
    torch.testing.assert_close(weights[0], weights[1])
    assert not torch.allclose(values[0], values[1])  # visual updates DO change


@pytest.mark.parametrize('flag', ['QUALITY_ENABLE', 'UNIQUENESS_ENABLE'])
def test_v1_rejects_accidentally_enabled_judgments(flag):
    model = _concat_model()
    model.cfg.FEW_SHOT.QUERY_CLASS_MATCHABILITY[flag] = True
    with pytest.raises(ValueError, match='OFF'):
        model._init_text_concat_st()


def test_chunk_checkpoint_outputs_and_gradients_match():
    serial, batched = _concat_model(), _concat_model(True)
    serial.text_concat_st.class_chunk_size = serial.text_concat_st.video_chunk_size = 1
    f, _, meta = _episode()
    a, b = text_concat_episode(serial, f, meta), text_concat_episode(batched, f, meta)
    torch.testing.assert_close(a['query_partial_q2s_logits'], b['query_partial_q2s_logits'])
    _loss(serial, a, meta).backward()
    _loss(batched, b, meta).backward()
    for (name, p), (_, q) in zip(serial.named_parameters(), batched.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize('mlp,readout', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('stochastic', [False, True])
def test_selective_checkpoint_preserves_outputs_gradients_and_rng(mlp, readout, stochastic):
    from trokens.models.common import DropPath
    plain = _concat_model().train()
    selective = _concat_model(mlp_checkpoint=mlp, readout_checkpoint=readout).train()
    f, _, meta = _episode()
    fa, fb = f.clone().requires_grad_(), f.clone().requires_grad_()
    if stochastic:
        for model in (plain, selective):
            model.pos_drop = nn.Dropout(.1)
            model.blocks[0].drop_path = DropPath(.2)
            for module in model.blocks[0].modules():
                if isinstance(module, nn.Dropout):
                    module.p = .15
    torch.manual_seed(123)
    a = text_concat_episode(plain, fa, meta)
    _loss(plain, a, meta).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(123)
    b = text_concat_episode(selective, fb, meta)
    _loss(selective, b, meta).backward()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('query_partial_q2s_logits', 'global_classification_logits', 'text_concat_spatial_weights'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    torch.testing.assert_close(fa.grad, fb.grad, atol=3e-6, rtol=3e-5)
    for (name, p), (_, q) in zip(plain.named_parameters(), selective.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize('option', ['mlp_checkpoint', 'readout_checkpoint'])
def test_full_and_selective_checkpoint_cannot_be_nested(option):
    with pytest.raises(ValueError, match='either full'):
        _concat_model(checkpoint=True, **{option: True})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cuda_selective_checkpoint_matches_uncheckpointed_amp(dtype):
    plain = _concat_model().cuda().train()
    selective = _concat_model(mlp_checkpoint=True, readout_checkpoint=True).cuda().train()
    f, _, meta = _episode('cuda')
    fa, fb = f.clone().requires_grad_(), f.clone().requires_grad_()
    outputs = []
    for model, features in ((plain, fa), (selective, fb)):
        with torch.autocast('cuda', dtype=dtype):
            out = text_concat_episode(model, features, meta)
            loss = _loss(model, out, meta)
        loss.backward()
        outputs.append(out)
    for key in ('query_partial_q2s_logits', 'global_classification_logits', 'text_concat_spatial_weights'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    torch.testing.assert_close(fa.grad, fb.grad, atol=1e-5, rtol=1e-5)
    for (name, p), (_, q) in zip(plain.named_parameters(), selective.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5, msg=name)


def test_checkpoint_architecture_boundary_and_roundtrip(tmp_path):
    model = _concat_model()
    with pytest.raises(ValueError, match='token-count'):
        validate_label_pre_temporal_checkpoint(model, _concat_model(spatial_tokens=32).state_dict())
    state = model.state_dict()
    validate_label_pre_temporal_checkpoint(model, state)
    assert model._text_concat_st_checkpoint_loaded
    with pytest.raises(ValueError, match='Text-concat'):
        validate_label_pre_temporal_checkpoint(model, {}, allow_init=True)
    with pytest.raises(ValueError, match='Text-concat'):
        validate_label_pre_temporal_checkpoint(_new_model(), state)
    missing = dict(state)
    missing.pop('blocks.0.attn.qkv.weight')
    with pytest.raises(ValueError, match='Incomplete'):
        validate_label_pre_temporal_checkpoint(model, missing)
    mismatch = dict(state)
    mismatch['text_concat_st.architecture_signature'] = torch.tensor([1, 1, 16, 1])
    with pytest.raises(ValueError, match='token-count'):
        validate_label_pre_temporal_checkpoint(model, mismatch)
    path = tmp_path / 'concat.pyth'
    torch.save({'model_state': state, 'epoch': 0}, path)
    restored = _concat_model()
    assert load_checkpoint(str(path), restored, data_parallel=False) == 0
    f, _, meta = _episode()
    torch.testing.assert_close(text_concat_episode(model, f, meta)['query_partial_q2s_logits'],
                               text_concat_episode(restored, f, meta)['query_partial_q2s_logits'])


def test_forward_dispatch_preserves_motion_and_avoids_raw_extraction(monkeypatch):
    model = _concat_model().eval()
    f, _, meta = _episode()
    model.cfg.MF.USE_BASE_POS_EMBED = False
    meta['pred_tracks'] = torch.randn(4, 3, 5, 2)
    meta['hod_feat'] = torch.randn_like(f)
    calls = []
    def sample(value, tracks, add_pt_pos_embed):
        calls.append(add_pt_pos_embed)
        return value
    monkeypatch.setattr(model, 'get_dinotxt_features', lambda x: f)
    monkeypatch.setattr(model, '_sample_point_features', sample)
    model.hod_motion_module = nn.Identity()
    class Motion(nn.Module):
        def forward(self, tracks, visibility):
            return torch.ones_like(f) * .2
    model.cross_motion_module = Motion()
    result = model({'video': f, 'metadata': meta})
    expected = text_concat_episode(model, f + meta['hod_feat'] + .2, meta)
    assert len(calls) == 1  # no separate Raw sampling
    torch.testing.assert_close(result['query_partial_q2s_logits'], expected['query_partial_q2s_logits'])


class _EpisodeModel(nn.Module):
    def __init__(self, selective=False):
        super().__init__()
        self.encoder = _concat_model(checkpoint=not selective, mlp_checkpoint=selective,
                                     readout_checkpoint=selective)

    def forward(self, f, meta):
        return text_concat_episode(self.encoder, f, meta)


def _ddp_worker(rank, init_file, selective):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + init_file,
                                        rank=rank, world_size=2, timeout=timedelta(seconds=50))
    try:
        model = torch.nn.parallel.DistributedDataParallel(_EpisodeModel(selective), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            f, _, meta = _episode()
            out = model(f + rank * .01, meta)
            _loss(model.module.encoder, out, meta).backward()
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('selective', [False, True])
def test_two_process_cpu_ddp_checkpoint(tmp_path, selective):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / 'ddp-init'), selective), nprocs=2, join=True)
