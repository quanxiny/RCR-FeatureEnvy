from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .fixed_splits import split_labels


@dataclass(frozen=True)
class DatasetFiles:
    directory: Path
    labels: Path
    graph_cache: Path
    vocab: Path


def dataset_files(data_root: Path, task: int) -> DatasetFiles:
    name = "Dataset_Class_Class" if task == 1 else "Dataset_Method_Class"
    directory = data_root / name
    return DatasetFiles(
        directory=directory,
        labels=directory / "new_labels.txt",
        graph_cache=directory / "allDataDict.json",
        vocab=directory / "vocabDict.json",
    )


class OriginalDatasetAdapter:
    def __init__(self, data_root: Path, task: int, split_seed: int = 100):
        self.task = task
        self.split_seed = split_seed
        self.files = dataset_files(data_root, task)
        self.label_lines = self.files.labels.read_text(encoding="utf-8").splitlines()
        with self.files.vocab.open(encoding="utf-8") as handle:
            self.vocab = json.load(handle)
        with self.files.graph_cache.open(encoding="utf-8") as handle:
            self.graphs = json.load(handle)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def fold(self, fold: int):
        return split_labels(self.label_lines, fold, self.split_seed)

    @staticmethod
    def item_key(label_line: str) -> str:
        fields = label_line.split()
        return "_".join(fields[:3])

    def graph_tensors(self, label_line: str, device):
        key = self.item_key(label_line)
        data = self.graphs.get(key)
        if data is None:
            return None
        return (
            torch.as_tensor(data["h1_index"], dtype=torch.long, device=device),
            torch.as_tensor(data["edge_index1"], dtype=torch.long, device=device),
            torch.as_tensor(data["h2_index"], dtype=torch.long, device=device),
            torch.as_tensor(data["edge_index2"], dtype=torch.long, device=device),
            int(data["itemlabel"]),
        )

