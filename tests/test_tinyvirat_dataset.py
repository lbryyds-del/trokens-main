from pathlib import Path

import numpy as np
import pytest

from trokens.config.defaults import get_cfg
from trokens.datasets.base_ds import _load_point_queries
from trokens.datasets.utils import _repair_decord_rgb_layout
from trokens.datasets.tinyvirat import (
    _split_tinyvirat_line,
    _tinyvirat_label_ids,
)
from trokens.models.pointformer import Pointformer


def test_tinyvirat_config_class_partition_and_paths():
    cfg = get_cfg()
    cfg.merge_from_file("configs/trokens/tinyvirat.yaml")

    assert cfg.MODEL.NUM_CLASSES == 21
    assert len(cfg.TRAIN.CLASS_NAME) == 12
    assert len(cfg.TEST.CLASS_NAME) == 21
    assert cfg.TEST.DROP_EMPTY_QUERY_ROWS is True
    assert cfg.DATA.PATH_TO_DATA_DIR == "/data3/duanwei/data/tinyviratv2-3"
    assert Path(cfg.DATA.PATH_TO_SPLIT_DIR, "train_few_shot.txt").exists()
    assert set(cfg.TEST.SEEN_LABELS).isdisjoint(cfg.TEST.NOVEL_LABELS)
    assert sorted(cfg.TEST.SEEN_LABELS + cfg.TEST.NOVEL_LABELS) == list(range(21))


def test_tinyvirat_split_and_multilabel_parser():
    prefix, relative = _split_tinyvirat_line(
        "train3_14//videos/activity_carrying-activity_standing/clip.with.dots.mp4"
    )
    assert prefix == "train3_14"
    assert Path(relative).stem == "clip.with.dots"
    ids = _tinyvirat_label_ids(
        "activity_carrying-activity_standing",
        {"activity_carrying": 3, "activity_standing": 8},
    )
    assert ids == [3, 8]


def test_tinyvirat_unknown_class_has_actionable_error():
    with pytest.raises(ValueError, match="TEST.CLASS_NAME"):
        _tinyvirat_label_ids("Opening-unknown_action", {"Opening": 0})


def test_point_queries_current_key_is_loaded():
    point_queries = _load_point_queries({"point_queries": [0, 3, 7]}, 3)

    assert point_queries.tolist() == [0, 3, 7]


def test_point_queries_count_must_match_tracks():
    with pytest.raises(ValueError, match="does not match pred_tracks"):
        _load_point_queries({"point_queries": [0, 3]}, 3)


@pytest.mark.parametrize("padded_channels", [4, 6])
def test_decord_row_padding_is_repaired_to_rgb(padded_channels):
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)
    padded_rows = np.zeros((1, 2, 3 * padded_channels), dtype=np.uint8)
    padded_rows[..., : 3 * 3] = rgb.reshape(1, 2, 3 * 3)
    padded_video = padded_rows.reshape(1, 2, 3, padded_channels)

    repaired = _repair_decord_rgb_layout(padded_video)

    assert repaired.shape == rgb.shape
    np.testing.assert_array_equal(repaired, rgb)


def test_standard_decord_rgb_layout_is_unchanged():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 3, 3)

    assert _repair_decord_rgb_layout(rgb) is rgb


def test_tinyvirat_uses_its_own_five_prompt_descriptions():
    cfg = get_cfg()
    cfg.merge_from_file("configs/trokens/tinyvirat.yaml")
    model = Pointformer.__new__(Pointformer)
    model.cfg = cfg
    model.num_classes = cfg.MODEL.NUM_CLASSES
    model.atomic_label_names = list(cfg.TEST.CLASS_NAME)

    groups = model._load_label_prompt_groups()

    assert len(groups) == 21
    assert groups == cfg.DATA.LABEL_PROMPTS
    assert all(len(group) == len(set(group)) == 5 for group in groups)
    assert [group[0] for group in groups] == [
        "opening something", "interacting with an object or person", "pulling something",
        "carrying an object", "entering a place or vehicle", "a vehicle is moving",
        "exiting a place or vehicle", "talking with another person", "a vehicle turns left",
        "sitting down or remaining seated", "closing something", "walking", "pushing something",
        "using a tool", "standing", "carrying a heavy object", "gesturing with the hands or arms",
        "a vehicle turns right", "talking on a phone", "texting on a phone", "a miscellaneous activity",
    ]
    assert all("student" not in prompt for group in groups for prompt in group)


def test_tinyvirat_empty_prompts_still_fall_back_to_its_class_names():
    cfg = get_cfg()
    cfg.merge_from_file("configs/trokens/tinyvirat.yaml")
    cfg.DATA.LABEL_PROMPTS = []
    model = Pointformer.__new__(Pointformer)
    model.cfg = cfg
    model.num_classes = cfg.MODEL.NUM_CLASSES
    model.atomic_label_names = list(cfg.TEST.CLASS_NAME)

    assert model._load_label_prompt_groups() == [
        [name.replace("_", " ")] for name in cfg.TEST.CLASS_NAME
    ]
