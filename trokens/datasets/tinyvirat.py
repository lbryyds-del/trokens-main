#!/usr/bin/env python3

"""TinyVIRAT multi-label few-shot dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import trokens.utils.logging as logging

from .base_ds import BaseDataset
from .build import DATASET_REGISTRY


logger = logging.get_logger(__name__)


def _split_tinyvirat_line(line):
    """Split ``train3_14//videos/class/video.mp4`` into its two parts."""
    line = line.strip()
    if not line:
        raise ValueError("Empty TinyVIRAT split line.")
    if "//" in line:
        return line.split("//", 1)
    return "", line


def _tinyvirat_label_ids(class_combo, class_to_id):
    """Map a hyphen-separated TinyVIRAT class combination to global ids."""
    class_names = [name.strip() for name in class_combo.split("-") if name.strip()]
    unknown = [name for name in class_names if name not in class_to_id]
    if unknown:
        raise ValueError(
            "Unknown TinyVIRAT class name(s): "
            f"{unknown}. Add them to TEST.CLASS_NAME in global label-id order."
        )
    return [class_to_id[name] for name in class_names]


def _multi_hot(label_ids, num_classes):
    """Create a TinyVIRAT multi-hot label vector."""
    label_vec = np.zeros(num_classes, dtype=np.float32)
    label_vec[label_ids] = 1.0
    return label_vec


@DATASET_REGISTRY.register()
class Tinyvirat(BaseDataset):
    """Read TinyVIRAT directly from ``{train,test}_few_shot.txt`` files."""

    def __init__(self, cfg, mode):
        super(Tinyvirat, self).__init__(cfg, mode)

    def _resolve_video_path(self, source_prefix, relative_path):
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            return str(relative_path)

        # Split entries retain a source directory such as train3_14. Also
        # accept a flattened data root for converted copies.
        candidates = []
        if source_prefix:
            candidates.append(Path(self.data_root) / source_prefix / relative_path)
        candidates.append(Path(self.data_root) / relative_path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _construct_loader(self):
        self.data_root = self.cfg.DATA.PATH_TO_DATA_DIR
        split_root = self.cfg.DATA.PATH_TO_SPLIT_DIR or self.data_root
        split_file = Path(split_root) / f"{self.mode}_few_shot.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing TinyVIRAT split file: {split_file}")

        class_names = list(self.cfg.TEST.CLASS_NAME)
        if len(class_names) != self.cfg.MODEL.NUM_CLASSES:
            raise ValueError(
                "TinyVIRAT requires TEST.CLASS_NAME to contain exactly "
                f"MODEL.NUM_CLASSES={self.cfg.MODEL.NUM_CLASSES} names; "
                f"got {len(class_names)}."
            )
        if len(set(class_names)) != len(class_names):
            raise ValueError("TinyVIRAT TEST.CLASS_NAME contains duplicate names.")
        class_to_id = {name: class_id for class_id, name in enumerate(class_names)}

        rows = []
        for line_no, raw_line in enumerate(
            split_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            source_prefix, relative_path = _split_tinyvirat_line(raw_line)
            path_parts = Path(relative_path).parts
            if len(path_parts) < 3 or path_parts[0] != "videos":
                raise ValueError(
                    f"Unexpected TinyVIRAT path at {split_file}:{line_no}: "
                    f"{raw_line}"
                )
            class_combo = path_parts[1]
            label_ids = _tinyvirat_label_ids(class_combo, class_to_id)
            video_path = self._resolve_video_path(source_prefix, relative_path)
            # Path.stem removes only the final extension. TinyVIRAT filenames
            # contain many dots, so split('.')[0] would corrupt feature names.
            video_name = Path(relative_path).stem
            rows.append(
                {
                    "video_path": video_path,
                    "vid_id": str(Path(source_prefix) / relative_path),
                    "video_name": video_name,
                    "atomic_label_ids": label_ids,
                    "label_id": _multi_hot(label_ids, len(class_names)),
                    "feat_path": os.path.join(
                        self.base_feature_path, f"{video_name}.pkl"
                    ),
                }
            )

        self.split_df = pd.DataFrame(rows)
        original_len = len(self.split_df)
        if original_len == 0:
            raise ValueError(f"TinyVIRAT split file is empty: {split_file}")

        self.split_df = self.split_df[
            self.split_df["video_path"].apply(os.path.exists)
        ].reset_index(drop=True)
        video_filtered_len = len(self.split_df)
        if video_filtered_len != original_len:
            logger.warning(
                "Filtered %s TinyVIRAT rows with missing local videos for split %s.",
                original_len - video_filtered_len,
                self.mode,
            )

        self.split_df = self.split_df[
            self.split_df["feat_path"].apply(os.path.exists)
        ].reset_index(drop=True)
        if len(self.split_df) != video_filtered_len:
            logger.warning(
                "Filtered %s TinyVIRAT rows with missing point features for split %s.",
                video_filtered_len - len(self.split_df),
                self.mode,
            )
        if len(self.split_df) == 0:
            raise FileNotFoundError(
                f"No TinyVIRAT rows left for split {self.mode}. Check videos under "
                f"{self.data_root} and point features under {self.base_feature_path}."
            )

        self._path_to_videos = []
        self._make_final_lists()
        self._atomic_labels_singles = self.split_df["atomic_label_ids"].tolist()
        self._atomic_labels = []
        for labels in self._atomic_labels_singles:
            for _ in range(self._num_clips):
                self._atomic_labels.append(tuple(labels))
