"""Frozen text caching preserves prompt semantics, precision and projection grads."""

import copy
from datetime import timedelta

import pytest
import torch
from torch import nn

from trokens.config.defaults import get_cfg
from trokens.models.pointformer import Pointformer
from test_text_concat_conditioning import _concat_model, _loss
from test_learned_query_verifier import _episode
from trokens.models.text_concat_conditioning import text_concat_episode


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompts):
        self.calls.append(tuple(prompts))
        return torch.tensor([[len(p) % 32, sum(map(ord, p)) % 32] for p in prompts])


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.5)
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        return self.projection(self.dropout(self.embedding(tokens).mean(1)))


def _attach_text(model, cache):
    model.feat_extractor_type = 'dinotxt_vitl14_reg4'
    model.use_text_concat_st = True
    model.cfg.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = cache
    model.atomic_label_prompt_groups = [['walk', 'person walking'], ['sit'], ['jump', 'jumping']]
    model.dinotxt_text_model = _Tower().requires_grad_(False)
    model.dinotxt_tokenizer = _Tokenizer()
    model.text_to_model_proj = nn.Linear(4, 4)
    return model


def _model(cache=True):
    torch.manual_seed(181)
    model = Pointformer.__new__(Pointformer)
    nn.Module.__init__(model)
    model.cfg = get_cfg()
    return _attach_text(model, cache)


def _text(model, ids=(0, 1), dtype=torch.float32):
    return model._get_pot_label_text_features(torch.tensor(ids), dtype)


def test_cached_text_matches_uncached_values_order_and_projection_gradients():
    cached, plain = _model(), _model(False)
    for ids in ((0, 1), (1, 0), (2, 0, 2), (2, 1, 0)):
        cached.zero_grad(set_to_none=True)
        plain.zero_grad(set_to_none=True)
        a, b = _text(cached, ids), _text(plain, ids)
        torch.testing.assert_close(a, b)
        coefficient = torch.arange(a.numel()).reshape_as(a)
        (a * coefficient).sum().backward()
        (b * coefficient).sum().backward()
        for p, q in zip(cached.text_to_model_proj.parameters(), plain.text_to_model_proj.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        assert all(p.grad is None for p in cached.dinotxt_text_model.parameters())
    assert cached.dinotxt_text_model.calls == 2
    assert plain.dinotxt_text_model.calls == 4
    assert cached.dinotxt_tokenizer.calls == [('walk', 'person walking', 'sit'), ('jump', 'jumping')]
    assert _text(cached, dtype=torch.float16).dtype == torch.float16
    assert cached.dinotxt_text_model.calls == 2
    assert all(value.grad_fn is None for value in cached._frozen_label_text_cache.features.values())
    assert not any('cache' in name for name in cached.state_dict())


def test_projection_updates_are_not_cached_and_have_gradients_on_every_hit():
    model = _model()
    first = _text(model).detach()
    optimizer = torch.optim.SGD(model.text_to_model_proj.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = _text(model)
        out[:, 0].sum().backward()
        assert model.text_to_model_proj.weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.allclose(first, _text(model))
    assert model.dinotxt_text_model.calls == 1


def test_prompt_tokenizer_and_tower_state_changes_invalidate_cached_values():
    model = _model()
    _text(model)
    model.atomic_label_prompt_groups[0][0] = 'new walking prompt'
    _text(model)
    assert model.dinotxt_text_model.calls == 2
    assert model.dinotxt_tokenizer.calls[-1] == ('new walking prompt', 'person walking')
    model.dinotxt_tokenizer = _Tokenizer()
    _text(model)
    assert model.dinotxt_text_model.calls == 3
    state = copy.deepcopy(model.dinotxt_text_model.state_dict())
    state['projection.bias'].add_(.3)
    model.dinotxt_text_model.load_state_dict(state)
    actual = _text(model)
    assert model.dinotxt_text_model.calls == 4
    model.cfg.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = False
    torch.testing.assert_close(actual, _text(model))
    assert not model._frozen_label_text_cache.features


def test_precision_switch_and_eval_to_train_do_not_reuse_wrong_precision():
    model = _model()
    with torch.inference_mode():
        _text(model.eval())
    _text(model.train()).sum().backward()
    assert model.dinotxt_text_model.calls == 1
    assert model.text_to_model_proj.weight.grad is not None
    assert not model.dinotxt_text_model.training
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = _text(model)
        _text(model)
    assert model.dinotxt_text_model.calls == 2
    model.cfg.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = False
    with torch.autocast('cpu', dtype=torch.bfloat16):
        torch.testing.assert_close(actual, _text(model), rtol=0, atol=0)
    model.cfg.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = True
    _text(model)
    assert model.dinotxt_text_model.calls == 4
    model.dinotxt_text_model.double()
    _text(model)
    assert model.dinotxt_text_model.calls == 5


@pytest.mark.parametrize('reason', ['trainable_tower', 'historical_path', 'old_serialized_config'])
def test_cache_bypassed_outside_frozen_concat_path(reason):
    model = _model()
    if reason == 'trainable_tower':
        model.dinotxt_text_model.requires_grad_(True)
    elif reason == 'old_serialized_config':
        model.cfg.FEW_SHOT.TEXT_CONCAT_ST.pop('CACHE_FROZEN_TEXT')
    else:
        model.use_text_concat_st = False
    a, b = _text(model), _text(model)
    torch.testing.assert_close(a, b)
    assert model.dinotxt_text_model.calls == 2


@pytest.mark.parametrize('checkpoint', [False, True])
def test_cached_episode_forward_and_backward_match_uncached(checkpoint):
    plain = _concat_model(checkpoint)
    del plain._get_pot_label_text_features  # use the real Pointformer method
    _attach_text(plain, False)
    cached = copy.deepcopy(plain)
    cached.cfg.FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT = True
    _text(cached)  # exercise the HIT path, not just the first encoding
    features, _, meta = _episode()
    a = text_concat_episode(plain, features, meta)
    b = text_concat_episode(cached, features, meta)
    for key in ('query_partial_q2s_logits', 'text_concat_spatial_weights', 'global_classification_logits'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    _loss(plain, a, meta).backward()
    _loss(cached, b, meta).backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), cached.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
    assert cached.dinotxt_text_model.calls == 1


class _CachedEpisodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _concat_model(True)
        del self.encoder._get_pot_label_text_features
        _attach_text(self.encoder, True)

    def forward(self, features, metadata):
        return text_concat_episode(self.encoder, features, metadata)


def _cache_ddp_worker(rank, init_file):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + init_file,
                                        rank=rank, world_size=2, timeout=timedelta(seconds=50))
    try:
        model = nn.parallel.DistributedDataParallel(_CachedEpisodeModel(), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            features, _, metadata = _episode()
            output = model(features + rank * .01, metadata)
            _loss(model.module.encoder, output, metadata).backward()
            optimizer.step()
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    reference = p.detach().clone()
                    torch.distributed.broadcast(reference, src=0)
                    torch.testing.assert_close(p, reference, rtol=0, atol=0)
        assert model.module.encoder.dinotxt_text_model.calls == 1
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_cpu_ddp_keeps_cache_local_and_parameters_synchronized(tmp_path):
    torch.multiprocessing.spawn(_cache_ddp_worker, args=(str(tmp_path / 'text-cache-ddp'),),
                                nprocs=2, join=True)
