from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .cglsmn_outputs import CGLSMNForwardOutput, local_attention, local_features


class CrossGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, 2 * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    @staticmethod
    def div_with_small_value(numerator, denominator, eps=1e-8):
        denominator = denominator * (denominator > eps) + eps * (denominator <= eps)
        return numerator / denominator

    def cosine_attention(self, first, second):
        products = torch.mm(first, second.permute(1, 0))
        first_norm = first.norm(p=2, dim=1, keepdim=True)
        second_norm = second.norm(p=2, dim=1, keepdim=True).permute(1, 0)
        return self.div_with_small_value(products, first_norm * second_norm)

    def forward(self, h1, h2, return_attention=False, return_local_features=False):
        wh1 = torch.relu(torch.mm(h1, self.W))
        wh2 = torch.relu(torch.mm(h2, self.W))
        similarity = torch.relu(self.cosine_attention(wh1, wh2))
        attention1 = torch.mean(similarity, dim=1).reshape(-1, 1)
        attention2 = torch.mean(similarity, dim=0).reshape(-1, 1)
        outputs = (h1 * attention1, h2 * attention2)
        if return_attention or return_local_features:
            details = (*outputs, similarity, attention1, attention2)
            if return_local_features:
                details = (*details, wh1, wh2)
            return details
        return outputs


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, inputs):
        h0 = inputs.new_zeros(self.layer_dim * 2, 1, self.hidden_dim)
        c0 = inputs.new_zeros(self.layer_dim * 2, 1, self.hidden_dim)
        _, (hidden, _) = self.lstm(inputs, (h0, c0))
        final = torch.cat([hidden[::2][-1], hidden[1::2][-1]], dim=-1)
        return self.fc(final)


class CGLSMNBaseline(nn.Module):
    """Stable wrapper preserving the published model structure and state keys."""

    def __init__(self, vocab_size, hidden=128, in_features=128, out_features=128,
                 class_num=2, dropout=0.1, alpha=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.GCN1 = GCNConv(hidden, hidden)
        self.GCN2 = GCNConv(hidden, hidden)
        self.readOut1 = CrossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        self.readOut2 = CrossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        self.readOut3 = CrossGraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
        self.bi_lstm1 = LSTMModel(hidden, 100, 2, hidden)
        self.bi_lstm2 = LSTMModel(hidden, 100, 2, hidden)
        self.bi_lstm3 = LSTMModel(hidden, 100, 2, hidden)
        self.fusionBiLSTM = LSTMModel(2 * hidden, 100, 2, hidden)
        self.fc = nn.Linear(hidden, class_num)

    def forward(self, h1_index, edge_index1, h2_index, edge_index2,
                return_features=False, return_attention=False,
                return_local_features=False):
        h1 = self.embed(h1_index)
        h2 = self.embed(h2_index)
        return self.forward_embeddings(
            h1, edge_index1, h2, edge_index2,
            return_features=return_features,
            return_attention=return_attention,
            return_local_features=return_local_features,
        )

    def forward_embeddings(
        self,
        h1,
        edge_index1,
        h2,
        edge_index2,
        edge_weight1=None,
        edge_weight2=None,
        return_features=False,
        return_attention=False,
        return_local_features=False,
    ):
        """Run CG-LSMN from continuous node embeddings.

        This preserves the published token-ID forward path while exposing the
        continuous inputs and intra-graph edge weights required by attribution
        and pair-graph explanation methods.
        """
        attention = {}
        local = {}
        if return_attention or return_local_features:
            details = self.readOut1(
                h1, h2, return_attention=return_attention,
                return_local_features=return_local_features)
            text_h1, text_h2, similarity, weights1, weights2 = details[:5]
            if return_attention:
                attention["text"] = local_attention(similarity, weights1, weights2)
            if return_local_features:
                local["text"] = local_features(details[5], details[6])
        else:
            text_h1, text_h2 = self.readOut1(h1, h2)
        h1 = F.relu(self.GCN1(h1, edge_index1, edge_weight=edge_weight1))
        h2 = F.relu(self.GCN1(h2, edge_index2, edge_weight=edge_weight2))
        if return_attention or return_local_features:
            details = self.readOut2(
                h1, h2, return_attention=return_attention,
                return_local_features=return_local_features)
            type_h1, type_h2, similarity, weights1, weights2 = details[:5]
            if return_attention:
                attention["type"] = local_attention(similarity, weights1, weights2)
            if return_local_features:
                local["type"] = local_features(details[5], details[6])
        else:
            type_h1, type_h2 = self.readOut2(h1, h2)
        h1 = F.relu(self.GCN2(h1, edge_index1, edge_weight=edge_weight1))
        h2 = F.relu(self.GCN2(h2, edge_index2, edge_weight=edge_weight2))
        if return_attention or return_local_features:
            details = self.readOut3(
                h1, h2, return_attention=return_attention,
                return_local_features=return_local_features)
            call_h1, call_h2, similarity, weights1, weights2 = details[:5]
            if return_attention:
                attention["call"] = local_attention(similarity, weights1, weights2)
            if return_local_features:
                local["call"] = local_features(details[5], details[6])
        else:
            call_h1, call_h2 = self.readOut3(h1, h2)

        text_graph1 = self.bi_lstm1(text_h1.unsqueeze(1))
        text_graph2 = self.bi_lstm1(text_h2.unsqueeze(1))
        type_graph1 = self.bi_lstm2(type_h1.unsqueeze(1))
        type_graph2 = self.bi_lstm2(type_h2.unsqueeze(1))
        call_graph1 = self.bi_lstm3(call_h1.unsqueeze(1))
        call_graph2 = self.bi_lstm3(call_h2.unsqueeze(1))
        text_info = torch.cat([text_graph1, text_graph2], dim=-1)
        type_info = torch.cat([type_graph1, type_graph2], dim=-1)
        call_info = torch.cat([call_graph1, call_graph2], dim=-1)
        fusion = self.fusionBiLSTM(torch.stack([text_info, type_info, call_info], dim=0))
        pair_embedding = fusion.reshape(1, -1)
        logits = self.fc(pair_embedding)
        probabilities = F.softmax(logits, dim=-1)
        if not return_features and not return_attention and not return_local_features:
            return probabilities

        output: CGLSMNForwardOutput = {
            "logits": logits,
            "probabilities": probabilities,
        }
        if return_features:
            output.update({
                "pair_embedding": pair_embedding,
                "graph1_embedding": torch.cat(
                    [text_graph1, type_graph1, call_graph1], dim=-1),
                "graph2_embedding": torch.cat(
                    [text_graph2, type_graph2, call_graph2], dim=-1),
            })
        if return_attention:
            output["attention"] = attention
        if return_local_features:
            output["local_features"] = local
        return output
