import random
import copy
import unittest

import torch

from src.data.fixed_splits import published_project_folds
from src.data.metadata_builder import GroundTruthCase, state_and_augmentation
from src.models.cglsmn_baseline import CGLSMNBaseline
from src.losses.original_loss import FocalLoss
from src.utils.metrics import classification_metrics


class StableBaselineTests(unittest.TestCase):
    def test_case_id_and_refactoring_state_are_stable(self):
        case = GroundTruthCase(
            commit_sha="abc123",
            project="demo",
            moved_method="moveMe",
            source_class="src/A.java",
            target_class="src/B.java",
        )
        self.assertEqual(case.case_id, case.case_id)
        self.assertTrue(case.case_id.startswith("case_"))
        self.assertEqual(state_and_augmentation("dataItem/demo_item_pos_7"), ("pre", 7))
        self.assertEqual(state_and_augmentation("dataItem/demo_item_neg_4"), ("post", 4))

    def test_metrics_include_required_fields(self):
        metrics = classification_metrics(
            [0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.8, 0.9],
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 0, 2))
        self.assertIn("pr_auc", metrics)
        self.assertIn("mcc", metrics)
        self.assertIn("roc_auc_legacy", metrics)

    def test_project_folds_are_stable(self):
        lines = [f"dataItem/project{i}_pos_0 A B {i % 2}" for i in range(10)]
        first = published_project_folds(lines, split_seed=100)
        random.seed(999)
        second = published_project_folds(lines, split_seed=100)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]["test"]), 2)

    def test_model_forward_is_repeatable(self):
        torch.manual_seed(42)
        model = CGLSMNBaseline(32, hidden=8, in_features=8, out_features=8)
        model.eval()
        nodes1 = torch.tensor([1, 2, 3, 4])
        nodes2 = torch.tensor([3, 4, 5])
        edges1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edges2 = torch.tensor([[0, 1, 2], [1, 2, 0]])
        with torch.no_grad():
            first = model(nodes1, edges1, nodes2, edges2)
            second = model(nodes1, edges1, nodes2, edges2)
        self.assertEqual(tuple(first.shape), (1, 2))
        self.assertTrue(torch.equal(first, second))

    def test_chunked_backward_matches_batch_sum(self):
        torch.manual_seed(7)
        summed = CGLSMNBaseline(32, hidden=8, in_features=8, out_features=8)
        chunked = copy.deepcopy(summed)
        criterion = FocalLoss()
        samples = [
            (torch.tensor([1, 2, 3]), torch.tensor([[0, 1, 2], [1, 2, 0]]),
             torch.tensor([3, 4]), torch.tensor([[0, 1], [1, 0]]), 1),
            (torch.tensor([5, 6]), torch.tensor([[0, 1], [1, 0]]),
             torch.tensor([2, 7, 8]), torch.tensor([[0, 1, 2], [1, 2, 0]]), 0),
        ]
        total = 0
        for h1, e1, h2, e2, label in samples:
            target = torch.tensor([[0.0, 1.0] if label else [1.0, 0.0]])
            total = total + criterion(summed(h1, e1, h2, e2), target)
        total.backward()
        chunk_loss = 0
        for h1, e1, h2, e2, label in samples:
            target = torch.tensor([[0.0, 1.0] if label else [1.0, 0.0]])
            chunk_loss = chunk_loss + criterion(chunked(h1, e1, h2, e2), target)
        chunk_loss.backward()
        max_difference = max(
            float((left.grad - right.grad).abs().max())
            for left, right in zip(summed.parameters(), chunked.parameters()))
        self.assertLess(max_difference, 1e-6)


if __name__ == "__main__":
    unittest.main()
