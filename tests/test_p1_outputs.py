import unittest

import torch

from src.models.cglsmn_baseline import CGLSMNBaseline


class ExtendedOutputTests(unittest.TestCase):
    def test_optional_features_and_attention_preserve_default_output(self):
        torch.manual_seed(11)
        model = CGLSMNBaseline(32, hidden=8, in_features=8, out_features=8)
        model.eval()
        graph1 = torch.tensor([1, 2, 3])
        edges1 = torch.tensor([[0, 1, 2], [1, 2, 0]])
        graph2 = torch.tensor([4, 5])
        edges2 = torch.tensor([[0, 1], [1, 0]])
        default = model(graph1, edges1, graph2, edges2)
        extended = model(
            graph1, edges1, graph2, edges2,
            return_features=True, return_attention=True,
            return_local_features=True)
        self.assertTrue(torch.equal(default, extended["probabilities"]))
        self.assertEqual(tuple(extended["logits"].shape), (1, 2))
        self.assertEqual(tuple(extended["pair_embedding"].shape), (1, 8))
        self.assertEqual(tuple(extended["graph1_embedding"].shape), (1, 24))
        self.assertEqual(tuple(extended["graph2_embedding"].shape), (1, 24))
        for level in ("text", "type", "call"):
            local = extended["attention"][level]
            self.assertEqual(tuple(local["similarity"].shape), (3, 2))
            self.assertEqual(tuple(local["graph1_weights"].shape), (3, 1))
            self.assertEqual(tuple(local["graph2_weights"].shape), (2, 1))
            features = extended["local_features"][level]
            self.assertEqual(tuple(features["graph1"].shape), (3, 16))
            self.assertEqual(tuple(features["graph2"].shape), (2, 16))

    def test_continuous_embeddings_and_edge_weights_are_differentiable(self):
        torch.manual_seed(13)
        model = CGLSMNBaseline(32, hidden=8, in_features=8, out_features=8)
        model.train()
        graph1 = torch.tensor([1, 2, 3])
        edges1 = torch.tensor([[0, 1, 2], [1, 2, 0]])
        graph2 = torch.tensor([4, 5])
        edges2 = torch.tensor([[0, 1], [1, 0]])
        embedding1 = model.embed(graph1).detach().requires_grad_(True)
        embedding2 = model.embed(graph2).detach().requires_grad_(True)
        weight1 = torch.ones(edges1.shape[1], requires_grad=True)
        weight2 = torch.ones(edges2.shape[1], requires_grad=True)
        output = model.forward_embeddings(
            embedding1,
            edges1,
            embedding2,
            edges2,
            edge_weight1=weight1,
            edge_weight2=weight2,
            return_features=True,
        )
        output["logits"][0, 1].backward()
        for value in (embedding1.grad, embedding2.grad, weight1.grad, weight2.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())

        default = model(graph1, edges1, graph2, edges2)
        continuous = model.forward_embeddings(
            model.embed(graph1), edges1, model.embed(graph2), edges2
        )
        self.assertTrue(torch.equal(default, continuous))


if __name__ == "__main__":
    unittest.main()
