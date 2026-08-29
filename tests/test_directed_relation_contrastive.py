import tempfile
import unittest
from pathlib import Path

import torch

from src.data.directed_anchors import (
    build_directed_anchor_index,
    class_field_names,
)
from src.losses.directed_relation_contrastive import (
    directed_method_class_contrastive_loss,
)


class DirectedRelationContrastiveTests(unittest.TestCase):
    def features(self):
        torch.manual_seed(19)
        return {
            level: {
                "graph1": torch.randn(4, 8, requires_grad=True),
                "graph2": torch.randn(5, 8, requires_grad=True),
            }
            for level in ("text", "type", "call")
        }

    def test_positive_loss_is_directed_and_finite(self):
        features = self.features()
        method_tokens = torch.tensor([1, 2, 3, 4])
        class_tokens = torch.tensor([8, 2, 9, 3, 10])
        informative = torch.ones(16, dtype=torch.bool)
        losses, stats = directed_method_class_contrastive_loss(
            features,
            method_tokens,
            class_tokens,
            torch.tensor([2, 3]),
            informative,
            1,
            mutual_threshold=1.01,
        )
        self.assertEqual(stats["directed_evidence_pairs"], 4)
        self.assertEqual(stats["directed_mutual_pairs"], 0)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(features["type"]["graph1"].grad)
        self.assertIsNotNone(features["type"]["graph2"].grad)

    def test_negative_loss_uses_one_pair_level_hinge(self):
        features = self.features()
        tokens1 = torch.tensor([1, 2, 3, 4])
        tokens2 = torch.tensor([5, 6, 7, 8, 9])
        informative = torch.ones(16, dtype=torch.bool)
        losses, stats = directed_method_class_contrastive_loss(
            features,
            tokens1,
            tokens2,
            torch.empty(0, dtype=torch.long),
            informative,
            0,
            negative_margin=-1.0,
            negative_topk=2,
        )
        self.assertEqual(stats["directed_evidence_pairs"], 0)
        self.assertGreater(stats["directed_negative_pair_matches"], 0)
        self.assertGreater(float(losses["negative"].detach()), 0.0)
        losses["negative"].backward()
        self.assertIsNotNone(features["call"]["graph1"].grad)

    def test_disabled_fallbacks_do_not_add_pairs_or_loss(self):
        features = self.features()
        informative = torch.ones(16, dtype=torch.bool)
        positive_losses, positive_stats = directed_method_class_contrastive_loss(
            features,
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([5, 6, 7, 8, 9]),
            torch.empty(0, dtype=torch.long),
            informative,
            1,
            enable_mutual=False,
        )
        self.assertEqual(positive_stats["directed_mutual_pairs"], 0)
        self.assertEqual(float(positive_losses["mutual"].detach()), 0.0)
        negative_losses, negative_stats = directed_method_class_contrastive_loss(
            features,
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([5, 6, 7, 8, 9]),
            torch.empty(0, dtype=torch.long),
            informative,
            0,
            enable_negative=False,
        )
        self.assertEqual(
            negative_stats["directed_negative_pair_matches"], 0
        )
        self.assertEqual(float(negative_losses["negative"].detach()), 0.0)

    def test_anchor_index_uses_accessed_target_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = root / "dataItem" / "case"
            item.mkdir(parents=True)
            method = item / "move.java"
            target = item / "Target.java"
            method.write_text(
                "void move() { target.balance = target.total(); helper(); }",
                encoding="utf-8",
            )
            target.write_text(
                "class Target {\n"
                "  private int balance;\n"
                "  public int total() { return balance; }\n"
                "}\n",
                encoding="utf-8",
            )
            (item / "dataItemInfo.json").write_text(
                '{"tagClassName":"Target","srcClassName":"Source",'
                '"MethodNamesInTag":["total"],"MethodNamesInSrc":["move"]}',
                encoding="utf-8",
            )
            line = "dataItem/case move Target 1"
            metadata = {
                line: {
                    "graph1_path": str(method),
                    "graph2_path": str(target),
                    "item_path": "dataItem/case",
                    "label": "1",
                }
            }
            anchors, audit = build_directed_anchor_index(
                metadata, {"balance": 3, "total": 4, "helper": 5}
            )
            self.assertEqual(anchors[line], (3, 4))
            self.assertEqual(audit["positive_anchor_coverage"], 1.0)

    def test_field_parser_excludes_method_locals(self):
        source = """
        class Target {
          private int field;
          public void method() {
            final int local = 1;
          }
        }
        """
        self.assertEqual(class_field_names(source), {"field"})


if __name__ == "__main__":
    unittest.main()
