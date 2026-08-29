import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.aggregate_continual import run_summary
from scripts.run_continual import uses_distillation, uses_ewc, uses_replay
from src.continual.training import (
    balanced_epoch_size,
    estimate_diagonal_fisher,
    ewc_penalty,
    rows_for_cases,
    relation_distillation_loss,
    select_replay_case_ids,
    train_epoch,
)
from src.losses.original_loss import FocalLoss
from src.utils.seed import restore_rng_state


def replay_rows(projects=10, cases_per_project=2):
    rows = []
    for project in range(projects):
        for case in range(cases_per_project):
            case_id = f"case_{project}_{case}"
            for label, state in (("0", "post"), ("1", "pre")):
                rows.append({
                    "case_id": case_id,
                    "split_project": f"project_{project}",
                    "label": label,
                    "state": state,
                })
    return rows


def metric_row(stage, increment, accuracy, split="probe"):
    return {
        "train_stage": str(stage),
        "eval_increment": str(increment),
        "eval_split": split,
        "accuracy": str(accuracy),
        "precision": "0.6", "recall": "0.7", "f1": "0.65",
        "roc_auc": "0.75", "roc_auc_legacy": "0.8", "pr_auc": "0.72",
        "mcc": "0.4", "tp": "7", "fp": "3", "tn": "8", "fn": "2",
        "samples": "20", "not_found": "0", "inference_seconds": "1.0",
    }


class ContinualAlgorithmTests(unittest.TestCase):
    def test_supcon_objective_is_optimized_within_a_continual_batch(self):
        class PairModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(1, 2)
                self.classifier = torch.nn.Linear(2, 2)

            def forward(self, h1, _e1, _h2, _e2, return_features=False, **_):
                pair = self.encoder(h1[:1])
                probabilities = torch.softmax(self.classifier(pair), dim=1)
                if return_features:
                    return {
                        "probabilities": probabilities,
                        "pair_embedding": pair,
                    }
                return probabilities

        class Adapter:
            @staticmethod
            def graph_tensors(line, device):
                label = int(line.split()[3])
                value = float(line.split()[1])
                node = torch.tensor([[value]], device=device)
                edge = torch.ones((2, 1), dtype=torch.long, device=device)
                return node, edge, node, edge, label

        model = PairModel()
        projection = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(projection.parameters()), lr=0.01
        )
        metrics = train_epoch(
            model,
            optimizer,
            FocalLoss(),
            Adapter(),
            ["item 0.0 a 0", "item 0.1 b 0", "item 1.0 c 1", "item 1.1 d 1"],
            batch_size=4,
            backward_chunk=4,
            device=torch.device("cpu"),
            rng=random.Random(42),
            projection=projection,
            supcon_options={"temperature": 0.07, "coefficient": 0.20},
        )
        self.assertEqual(metrics["train_samples"], 4)
        self.assertGreater(metrics["supervised_contrastive_positive_pairs"], 0)
        self.assertGreater(metrics["supervised_contrastive_negative_pairs"], 0)
        self.assertGreaterEqual(metrics["supervised_contrastive_loss"], 0.0)

    def test_replay_relation_method_flags_are_composable(self):
        self.assertTrue(uses_replay("replay_relation_ewc"))
        self.assertTrue(uses_distillation("replay_relation_ewc"))
        self.assertTrue(uses_ewc("replay_relation_ewc"))
        self.assertFalse(uses_ewc("replay_relation"))
        self.assertFalse(uses_distillation("replay"))

    def test_relation_distillation_is_zero_for_identical_states(self):
        attention = {
            level: {
                "graph1_weights": torch.tensor([0.2, 0.8]),
                "graph2_weights": torch.tensor([0.6, 0.4]),
            }
            for level in ("text", "type", "call")
        }
        student = {
            "logits": torch.tensor([[0.2, 0.8]], requires_grad=True),
            "pair_embedding": torch.tensor([[1.0, 0.0]], requires_grad=True),
            "attention": attention,
        }
        teacher = {
            "logits": student["logits"].detach().clone(),
            "pair_embedding": student["pair_embedding"].detach().clone(),
            "attention": {
                level: {
                    name: value.detach().clone()
                    for name, value in values.items()
                }
                for level, values in attention.items()
            },
        }
        loss, components = relation_distillation_loss(
            student,
            teacher,
            logit_coefficient=0.5,
            embedding_coefficient=0.1,
            attention_coefficient=0.1,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertEqual(
            set(components),
            {
                "logit_distillation_loss",
                "embedding_distillation_loss",
                "attention_distillation_loss",
                "effective_distillation_loss",
            },
        )

    def test_balanced_epoch_budget_matches_original_sampler(self):
        lines = [f"item a b {label}" for label in ([1] * 5 + [0] * 9)]
        self.assertEqual(balanced_epoch_size(lines, batch_size=2), 10)
        self.assertEqual(balanced_epoch_size(lines, batch_size=4), 8)

    def test_replay_selects_complete_cases_deterministically(self):
        rows = replay_rows()
        first = select_replay_case_ids(rows, ratio=0.10, seed=42)
        second = select_replay_case_ids(rows, ratio=0.10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        selected_rows = rows_for_cases(rows, first)
        self.assertEqual({row["case_id"] for row in selected_rows}, set(first))
        for case_id in first:
            case_rows = [row for row in selected_rows if row["case_id"] == case_id]
            self.assertEqual({row["label"] for row in case_rows}, {"0", "1"})
            self.assertEqual({row["state"] for row in case_rows}, {"pre", "post"})
        self.assertEqual(len({row["split_project"] for row in selected_rows}), 2)

    def test_replay_prefers_complete_cases_within_a_project(self):
        rows = replay_rows(projects=1, cases_per_project=10)
        rows = [
            row for row in rows
            if not (row["case_id"] == "case_0_0" and row["label"] == "1")
        ]
        selected = select_replay_case_ids(rows, ratio=0.10, seed=1)
        self.assertEqual(len(selected), 1)
        self.assertNotEqual(selected[0], "case_0_0")

    def test_forgetting_and_backward_transfer_follow_specification(self):
        config = {
            "run_name": "demo", "method": "naive", "task": 1,
            "fold": 1, "seed": 42, "stage_epochs": 8, "buffer_ratio": None,
        }
        rows = [
            metric_row(1, 1, 0.80),
            metric_row(2, 1, 0.70), metric_row(2, 2, 0.75),
            metric_row(3, 1, 0.60), metric_row(3, 2, 0.70),
            metric_row(3, 3, 0.80), metric_row(3, 0, 0.77, "outer_test"),
        ]
        summary, matrix = run_summary(config, rows)
        self.assertAlmostEqual(summary["average_forgetting"], 0.125)
        self.assertAlmostEqual(summary["backward_transfer"], -0.125)
        self.assertAlmostEqual(summary["final_average_probe_accuracy"], 0.70)
        self.assertEqual(summary["material_forgetting"], 1)
        self.assertEqual(len(matrix), 6)
        replay_summary, _ = run_summary(
            {**config, "run_name": "replay", "method": "replay"}, rows
        )
        self.assertEqual(replay_summary["material_forgetting"], 1)

    def test_ewc_penalty_is_zero_at_anchor_and_positive_after_change(self):
        model = torch.nn.Linear(2, 1, bias=False)
        anchor = {name: value.detach().clone() for name, value in model.named_parameters()}
        fisher = {name: torch.ones_like(value) for name, value in model.named_parameters()}
        consolidation = [{
            "stage": 1, "samples": 4, "anchor": anchor, "fisher": fisher,
        }]
        self.assertEqual(float(ewc_penalty(model, consolidation, 10.0).detach()), 0.0)
        with torch.no_grad():
            model.weight.add_(0.5)
        self.assertAlmostEqual(
            float(ewc_penalty(model, consolidation, 10.0).detach()), 5.0,
            places=6,
        )

    def test_fisher_temporarily_enables_training_mode_for_backward(self):
        class TrackingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 2)
                self.forward_training_modes = []

            def forward(self, h1, _e1, _h2, _e2):
                self.forward_training_modes.append(self.training)
                return torch.softmax(self.linear(h1[:1]), dim=1)

        class Adapter:
            @staticmethod
            def graph_tensors(line, device):
                label = int(line.split()[3])
                node = torch.ones((1, 1), device=device)
                edge = torch.ones((2, 1), dtype=torch.long, device=device)
                return node, edge, node, edge, label

        model = TrackingModel()
        model.eval()
        fisher, count, _ = estimate_diagonal_fisher(
            model,
            Adapter(),
            ["item a b 0", "item c d 1"],
            torch.device("cpu"),
            max_samples=2,
            seed=42,
        )
        self.assertEqual(count, 2)
        self.assertTrue(all(model.forward_training_modes))
        self.assertFalse(model.training)
        self.assertGreater(sum(float(value.sum()) for value in fisher.values()), 0.0)

    def test_rng_restore_moves_map_location_state_back_to_cpu(self):
        sampling_rng = random.Random(7)
        cpu_state = torch.get_rng_state()
        moved_state = Mock()
        moved_state.cpu.return_value = cpu_state
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": moved_state,
            "sampling": sampling_rng.getstate(),
        }
        with patch("src.utils.seed.torch.cuda.is_available", return_value=False):
            restore_rng_state(state, sampling_rng)
        moved_state.cpu.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
