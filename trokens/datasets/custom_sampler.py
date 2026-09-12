import numpy as np
import random
import torch
from torch.utils.data import Sampler

import trokens.utils.distributed as du

class FewShotEpisodeSampler(Sampler):
    def __init__(self, dataset, cfg, mode, less_iters=False):
        self.cfg = cfg
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        self.mode = mode
        self.rank = du.get_rank()
        self.world_size = max(du.get_world_size(), 1)
        self.base_seed = int(cfg.RNG_SEED)
        self.epoch = 0
        self.multi_label = cfg.DATA.MULTI_LABEL and hasattr(dataset, "_atomic_labels")
        labels = dataset._labels
        if self.multi_label:
            self.atomic_labels = [set(labels) for labels in dataset._atomic_labels]
            self.label_combos = [
                tuple(sorted(labels)) for labels in dataset._atomic_labels
            ]
            self.class_ids = sorted(
                {class_id for label_set in self.atomic_labels for class_id in label_set}
            )
        else:
            self.atomic_labels = None
            self.label_combos = None
            self.class_ids = list(np.unique(labels))
        self.num_way = cfg.FEW_SHOT.N_WAY
        self.num_support = cfg.FEW_SHOT.K_SHOT
        self.num_queries = (cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS if mode == 'train'
                                            else cfg.FEW_SHOT.TEST_QUERY_PER_CLASS)
        self.samples_per_class = self.num_support + self.num_queries
        self.batch_size = (self.num_way * self.samples_per_class)

        # Create a list of indices for each class.
        if self.multi_label:
            self.combo_indices = {}
            for idx, label_combo in enumerate(self.label_combos):
                self.combo_indices.setdefault(label_combo, []).append(idx)
            self.class_label_combos = {
                class_label: [
                    label_combo for label_combo in self.combo_indices
                    if class_label in label_combo
                ]
                for class_label in self.class_ids
            }
            self.class_indices = {
                class_label: [
                    idx for idx, label_set in enumerate(self.atomic_labels)
                    if class_label in label_set
                ]
                for class_label in self.class_ids
            }
        else:
            self.class_indices = {
                class_label: [
                    idx for idx, label in enumerate(labels) if label == class_label
                ]
                for class_label in self.class_ids
            }
        self.less_iters = less_iters
        self.local_episode_ids = self._build_local_episode_ids()

    def _episode_label(self, sample_idx, selected_classes):
        label_set = self.atomic_labels[sample_idx]
        return np.array(
            [1.0 if class_id in label_set else 0.0 for class_id in selected_classes],
            dtype=np.float32,
        )

    def _sample_indices_for_class(self, class_label, num_samples, used_indices, rng):
        candidates = list(self.class_indices[class_label])
        fresh_candidates = [idx for idx in candidates if idx not in used_indices]
        pool = fresh_candidates if len(fresh_candidates) >= num_samples else candidates
        if len(pool) >= num_samples:
            return rng.sample(pool, num_samples)
        return [rng.choice(pool) for _ in range(num_samples)]

    def _sample_indices_for_label_combo(
        self, class_label, num_samples, used_indices, rng
    ):
        """Sample support/query from one full label combo containing class_label."""
        label_combos = list(self.class_label_combos[class_label])
        fresh_label_combos = [
            label_combo for label_combo in label_combos
            if len([
                idx for idx in self.combo_indices[label_combo]
                if idx not in used_indices
            ]) >= num_samples
        ]
        eligible_label_combos = [
            label_combo for label_combo in label_combos
            if len(self.combo_indices[label_combo]) >= num_samples
        ]
        if fresh_label_combos:
            label_combo = rng.choice(fresh_label_combos)
            pool = [
                idx for idx in self.combo_indices[label_combo]
                if idx not in used_indices
            ]
        elif eligible_label_combos:
            label_combo = rng.choice(eligible_label_combos)
            pool = list(self.combo_indices[label_combo])
        else:
            label_combo = rng.choice(label_combos)
            pool = list(self.combo_indices[label_combo])
            fresh_pool = [idx for idx in pool if idx not in used_indices]
            if fresh_pool:
                pool = fresh_pool

        if len(pool) >= num_samples:
            return rng.sample(pool, num_samples)
        return [rng.choice(pool) for _ in range(num_samples)]

    def _total_episodes(self):
        if self.mode == 'train':
            return self.cfg.FEW_SHOT.TRAIN_EPISODES
        total = self.cfg.FEW_SHOT.TEST_EPISODES
        if self.less_iters:
            total = total // 5
        return total

    def _build_local_episode_ids(self):
        total = self._total_episodes()
        if self.mode == 'train' and self.cfg.FEW_SHOT.TRAIN_OG_EPISODES:
            return list(range(total))
        episode_ids = list(range(total))
        remainder = total % self.world_size
        if remainder:
            pad = self.world_size - remainder
            episode_ids.extend(episode_ids[:pad])
        return episode_ids[self.rank::self.world_size]

    def set_epoch(self, epoch):
        """Change training episodes without changing validation/test episodes."""
        self.epoch = int(epoch)

    def __iter__(self):
        for global_episode_idx in self.local_episode_ids:
            epoch_offset = (
                self.epoch * self.cfg.FEW_SHOT.TRAIN_EPISODES
                if self.mode == "train" else 0
            )
            rng = random.Random(self.base_seed + epoch_offset + global_episode_idx)
            selected_classes = rng.sample(self.class_ids, self.num_way)

            batch_indices = []
            sample_types = []
            batch_label = []
            episode_class_ids = []
            used_indices = set()

            sample_type = (['support'] * self.num_support +
                                            ['query'] * self.num_queries)
            for idx, class_label in enumerate(selected_classes):
                # Multi-label episodes first bind the class to one full label
                # combo, then sample support/query from that combo.
                if self.multi_label:
                    class_indices = self._sample_indices_for_label_combo(
                        class_label, self.samples_per_class, used_indices, rng
                    )
                    used_indices.update(class_indices)
                else:
                    class_indices = rng.sample(
                        self.class_indices[class_label],
                        self.samples_per_class,
                    )
                batch_indices.extend(class_indices)
                sample_types.extend(sample_type)
                if self.multi_label:
                    batch_label.extend([
                        self._episode_label(sample_idx, selected_classes)
                        for sample_idx in class_indices
                    ])
                    episode_class_ids.extend([
                        np.array(selected_classes, dtype=np.int64)
                        for _ in class_indices
                    ])
                else:
                    batch_label.extend([idx] * self.samples_per_class)
            batch_indices = np.array(batch_indices)
            sample_types = np.array(sample_types)
            batch_label = np.array(batch_label)
            episode_class_ids = np.array(episode_class_ids)
            indices = list(range(len(batch_indices)))

            # Shuffle the batch indices to mix the classes
            rng.shuffle(indices)
            batch_indices = batch_indices[indices]
            sample_types = sample_types[indices]
            batch_label = batch_label[indices]
            if self.multi_label:
                episode_class_ids = episode_class_ids[indices]
                index_and_sample_info = list(
                    zip(batch_indices, batch_label, sample_types, episode_class_ids)
                )
            else:
                index_and_sample_info = list(zip(batch_indices, batch_label, sample_types))

            yield index_and_sample_info

    def __len__(self):
        return len(self.local_episode_ids)
