import random
import unittest

import torch

from src.data.contrastive_sampler import ContrastiveBatchSampler, ContrastiveSample
from src.data.fixed_splits import inner_project_validation, project_name
from src.losses.case_contrastive import (
    case_aware_contrastive_loss,
    pre_post_hard_contrastive_loss,
)
from src.losses.supervised_contrastive import supervised_contrastive_loss


class ContrastiveLearningTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
            [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
        ], requires_grad=True)
        self.labels = torch.tensor([1, 1, 0, 0, 0, 1])
        self.case_ids = ["a", "a", "a", "a", "b", "b"]
        self.states = ["pre", "pre", "post", "post", "post", "pre"]
        self.canonical = [f"s{i}" for i in range(6)]

    def test_losses_have_pairs_finite_values_and_gradients(self):
        label_loss, label_stats = supervised_contrastive_loss(
            self.embeddings, self.labels, canonical_ids=self.canonical)
        hard_loss, hard_stats = pre_post_hard_contrastive_loss(
            self.embeddings, self.case_ids, self.states, self.labels, self.canonical)
        case_loss, case_stats = case_aware_contrastive_loss(
            self.embeddings, self.case_ids, self.states, self.labels, self.canonical)
        total = label_loss + hard_loss + case_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(self.embeddings.grad).all())
        self.assertGreater(label_stats["label_positive_pairs"], 0)
        self.assertGreater(hard_stats["pre_post_hard_negative_pairs"], 0)
        self.assertGreater(case_stats["same_case_same_state_positive_pairs"], 0)

    def test_exact_duplicate_graphs_are_not_contrastive_pairs(self):
        duplicated = ["same", "same"]
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 1])
        loss, stats = supervised_contrastive_loss(
            embeddings, labels, canonical_ids=duplicated)
        self.assertEqual(stats["label_positive_pairs"], 0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_sampler_preserves_epoch_multiset_and_exposes_hard_pairs(self):
        samples = [
            ContrastiveSample("a_pre_1", 1, "a", "pre", "a1"),
            ContrastiveSample("a_pre_2", 1, "a", "pre", "a2"),
            ContrastiveSample("a_post_1", 0, "a", "post", "a3"),
            ContrastiveSample("a_post_2", 0, "a", "post", "a4"),
            ContrastiveSample("b_pre_1", 1, "b", "pre", "b1"),
            ContrastiveSample("b_pre_2", 1, "b", "pre", "b2"),
            ContrastiveSample("b_post_1", 0, "b", "post", "b3"),
            ContrastiveSample("b_post_2", 0, "b", "post", "b4"),
        ]
        batches = list(ContrastiveBatchSampler(samples, 4, random.Random(42)))
        observed = sorted(sample.line for batch in batches for sample in batch)
        self.assertEqual(observed, sorted(sample.line for sample in samples))
        for batch in batches:
            self.assertGreaterEqual(sum(sample.label == 1 for sample in batch), 2)
            self.assertGreaterEqual(sum(sample.label == 0 for sample in batch), 2)
            self.assertTrue(any(
                left.case_id == right.case_id and left.state != right.state
                for i, left in enumerate(batch) for right in batch[i + 1:]))

    def test_inner_validation_is_project_disjoint_and_repeatable(self):
        lines = [
            f"dataItem/project{i}_case_item_pos_0 A B 1"
            for i in range(20) for _ in range(2)
        ]
        first = inner_project_validation(lines, task=1, fold=1)
        second = inner_project_validation(lines, task=1, fold=1)
        self.assertEqual(first, second)
        train_projects = {project_name(line) for line in first[0]}
        validation_projects = {project_name(line) for line in first[1]}
        self.assertFalse(train_projects & validation_projects)
        self.assertEqual(validation_projects, first[2])


if __name__ == "__main__":
    unittest.main()
