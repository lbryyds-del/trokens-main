import torch
from tools.probe_motion_counterfactual import valid_displacement_mask, shuffle_pair, condition_motion


def test_valid_displacement_requires_both_ends_and_no_first_frame():
    visible = torch.tensor([[[1, 0], [1, 1], [0, 1]]])
    expected = torch.tensor([[[0, 0], [1, 0], [0, 1]]], dtype=torch.bool)
    assert torch.equal(valid_displacement_mask(visible), expected)


def test_shuffle_is_paired_and_does_not_modify_inputs():
    h = torch.arange(2*8*3*4).reshape(2, 8, 3, 4).float()
    c = h + 1000
    original = h.clone()
    a, b = shuffle_pair(h, c, 9)
    assert torch.equal(a+1000, b)
    assert torch.equal(a.sort(dim=1).values, h.sort(dim=1).values)
    assert not torch.equal(a, h)
    assert torch.equal(h, original)
    assert torch.equal(shuffle_pair(h, c, 9)[0], a)


def test_conditions_keep_metadata_and_distinguish_null_from_disabled():
    class Dummy:
        def hod_motion_module(self, hist):
            return hist.permute(0, 2, 1, 3) + 2
        def cross_motion_module(self, tracks, visibility):
            visibility.zero_()  # Deliberately imitate a mutating encoder.
            return tracks + 3
    meta = {"pred_tracks": torch.rand(1, 8, 3, 2),
            "pred_visibility": torch.ones(1, 8, 3),
            "hod_feat": torch.ones(1, 3, 8, 2)}
    before = {k:v.clone() for k,v in meta.items()}
    h = torch.ones(1, 8, 3, 2)
    c = h * 2
    conditions = condition_motion(Dummy(), meta, h, c, 1)
    for key in meta:
        assert torch.equal(before[key], meta[key])
    assert conditions['motion_off'][0].count_nonzero() == 0
    assert conditions['null_inputs'][0].min() == 2
    assert conditions['static_geometry'][1].diff(dim=1).count_nonzero() == 0
    assert conditions['validity_masked'][0][:, 0].count_nonzero() == 0
