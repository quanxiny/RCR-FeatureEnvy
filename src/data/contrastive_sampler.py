from __future__ import annotations

import csv
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContrastiveSample:
    line: str
    label: int
    case_id: str
    state: str
    canonical_sample_id: str


def load_metadata_index(path: Path, task: int, fold: int,
                        split: str = "train") -> dict[str, dict]:
    index = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (int(row["task"]) == task and int(row["fold"]) == fold
                    and row["split"] == split):
                index.setdefault(row["label_line"], row)
    return index


def as_contrastive_samples(lines: list[str], metadata: dict[str, dict]) -> list[ContrastiveSample]:
    samples = []
    for line in lines:
        row = metadata.get(line)
        if row is None:
            raise KeyError(f"sample missing from metadata index: {line}")
        samples.append(ContrastiveSample(
            line=line,
            label=int(row["label"]),
            case_id=row["case_id"],
            state=row["state"],
            canonical_sample_id=row["canonical_sample_id"],
        ))
    return samples


class ContrastiveBatchSampler:
    """Deterministically reorder, but never add or remove, epoch samples."""

    def __init__(self, samples: list[ContrastiveSample], batch_size: int,
                 rng: random.Random):
        if batch_size < 4:
            raise ValueError("contrastive batch_size must be at least 4")
        if len(samples) % batch_size:
            raise ValueError("sample count must be divisible by batch_size")
        self.samples = samples
        self.batch_size = batch_size
        self.rng = rng

    def __iter__(self):
        remaining = set(range(len(self.samples)))
        all_indices = list(range(len(self.samples)))
        self.rng.shuffle(all_indices)
        all_pool = deque(all_indices)
        label_pools = defaultdict(list)
        case_state_pools = defaultdict(list)
        for index, sample in enumerate(self.samples):
            label_pools[sample.label].append(index)
            if sample.case_id:
                case_state_pools[(sample.case_id, sample.state)].append(index)
        for pool in [*label_pools.values(), *case_state_pools.values()]:
            self.rng.shuffle(pool)
        label_pools = {key: deque(value) for key, value in label_pools.items()}
        case_state_pools = {
            key: deque(value) for key, value in case_state_pools.items()}
        case_ids = sorted({key[0] for key in case_state_pools})
        strong_keys = sorted(
            key for key, pool in case_state_pools.items() if len(pool) >= 2)
        self.rng.shuffle(case_ids)
        self.rng.shuffle(strong_keys)

        def take(pool):
            while pool and pool[0] not in remaining:
                pool.popleft()
            if not pool:
                return None
            index = pool.popleft()
            remaining.remove(index)
            return index

        while remaining:
            batch = []
            # Prefer one genuine pre/post pair from the same refactoring case.
            for case_id in case_ids:
                pre = case_state_pools.get((case_id, "pre"), deque())
                post = case_state_pools.get((case_id, "post"), deque())
                pre_index = take(pre)
                if pre_index is None:
                    continue
                post_index = take(post)
                batch.append(pre_index)
                if post_index is not None:
                    batch.append(post_index)
                break

            # Prefer two distinct augmentations from the same case and state.
            # A same-state pair has one label; reserve at least one additional
            # slot so a hard pre/post pair can still be balanced to 2+2.
            if len(batch) <= self.batch_size - 3:
                for key in strong_keys:
                    pool = case_state_pools[key]
                    first = take(pool)
                    if first is None:
                        continue
                    second = take(pool)
                    batch.append(first)
                    if second is not None:
                        batch.append(second)
                    break

            # Guarantee label-level positives whenever the remaining set allows it.
            for label in (1, 0):
                while (sum(self.samples[index].label == label for index in batch) < 2
                       and len(batch) < self.batch_size):
                    index = take(label_pools.get(label, deque()))
                    if index is None:
                        break
                    batch.append(index)

            while len(batch) < self.batch_size:
                index = take(all_pool)
                if index is None:
                    break
                batch.append(index)
            if len(batch) != self.batch_size:
                raise RuntimeError("contrastive sampler exhausted before a full batch")
            yield [self.samples[index] for index in batch]


def count_contrastive_relations(samples: list[ContrastiveSample]) -> dict[str, int]:
    counts = {
        "label_positive_pairs": 0,
        "ordinary_negative_pairs": 0,
        "same_case_same_state_positive_pairs": 0,
        "pre_post_hard_negative_pairs": 0,
    }
    for index, left in enumerate(samples):
        for right in samples[index + 1:]:
            if left.canonical_sample_id == right.canonical_sample_id:
                continue
            same_case = bool(left.case_id) and left.case_id == right.case_id
            if left.label == right.label:
                counts["label_positive_pairs"] += 1
                if same_case and left.state == right.state:
                    counts["same_case_same_state_positive_pairs"] += 1
            elif same_case and left.state != right.state:
                counts["pre_post_hard_negative_pairs"] += 1
            elif not same_case:
                counts["ordinary_negative_pairs"] += 1
    return counts
