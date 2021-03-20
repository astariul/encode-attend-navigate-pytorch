import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib


LARGE_NUMBER = 100_000_000


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=True):
        """ Module for embedding the input features.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            batch_norm (bool, optional): Wether to apply batch normalization or
                not. Defaults to True.
        """
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = (
            nn.BatchNorm1d(out_dim, eps=0.001, momentum=0.01)
            if batch_norm
            else nn.Identity()
        )   # (Use TF default parameter, for consistency with original code)

    def forward(self, x):
        # x : [bs, seq, in_dim]
        emb_x = self.dense(x)   # [bs, out_dim, seq]
        return self.batch_norm(emb_x.transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(
        self, num_layers=3, n_hidden=512, ff_hidden=2048, num_heads=16, p_dropout=0.1
    ):
        """ Encoder layer

        Args:
            num_layers (int, optional): Number of layer in the Encoder. Defaults to 3.
            ff_hidden (int, optional): Size for the hidden layer of FF. Defaults to 2048.
            n_hidden (int, optional): Hidden size. Defaults to 512.
            num_heads (int, optional): Number of Attention heads. Defaults to 16.
            p_dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=num_heads, dim_feedforward=ff_hidden, dropout=p_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_seq):
        return self.transformer(input_seq)


class Pointer(nn.Module):
    def __init__(self, query_dim=256, n_hidden=512):
        """ Pointer network.

        Args:
            query_dim (int, optional): Dimension of the query. Defaults to 256.
            n_hidden (int, optional): Hidden size. Defaults to 512.
        """
        super().__init__()
        self.w_q = nn.Linear(query_dim, n_hidden, bias=False)
        self.v = nn.Parameter(torch.Tensor(n_hidden))
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_q.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.v, -bound, bound)    # Similar to a bias of Linear layer

    def forward(self, encoded_ref, query, mask, c=10, temp=1):
        encoded_query = self.w_q(query).unsqueeze(1)
        scores = torch.sum(self.v * torch.tanh(encoded_ref + encoded_query), dim=-1)
        scores = c * torch.tanh(scores / temp)
        masked_scores = torch.clip(
            scores - LARGE_NUMBER * mask, -LARGE_NUMBER, LARGE_NUMBER
        )
        return masked_scores


class FullGlimpse(nn.Module):
    def __init__(self, in_dim=128, out_dim=256):
        """ Full Glimpse for the Critic.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        """
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)
        self.v = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.v, -bound, bound)    # Similar to a bias of Linear layer

    def forward(self, ref):
        # Attention
        encoded_ref = self.dense(ref)
        scores = torch.sum(self.v * torch.tanh(encoded_ref), dim=-1)
        attention = F.softmax(scores, dim=-1)

        # Glimpse : Linear combination of reference vectors (define new query vector)
        glimpse = ref * attention.unsqueeze(-1)
        glimpse = torch.sum(glimpse, dim=1)
        return glimpse


class Decoder(nn.Module):
    def __init__(self, n_hidden=512, att_dim=256, query_dim=360, n_history=3):
        """ Decoder with a Pointer network and a memory of size `n_history`. 

        Args:
            n_hidden (int, optional): Encoder hidden size. Defaults to 512.
            att_dim (int, optional): Attention dimension size. Defaults to 256.
            query_dim (int, optional): Dimension of the query. Defaults to 360.
            n_history (int, optional): Size of history. Defaults to 3.
        """
        super().__init__()
        self.dense = nn.Linear(n_hidden, att_dim, bias=False)
        self.n_history = n_history
        self.queriers = nn.ModuleList([
            nn.Linear(n_hidden, query_dim, bias=False) for _ in range(n_history)
        ])
        self.pointer = Pointer(query_dim, att_dim)

    def forward(self, inputs, c=10, temp=1):
        batch_size, seq_len, hidden = inputs.size()

        idx_list, log_probs, entropies = [], [], []  # Tours index, log_probs, entropies
        mask = torch.zeros([batch_size, seq_len], device=inputs.device)  # Mask for actions

        encoded_input = self.dense(inputs)

        prev_actions = [torch.zeros([batch_size, hidden], device=inputs.device) for _ in range(self.n_history)]

        for _ in range(seq_len):
            query = F.relu(
                torch.stack(
                    [
                        querier(prev_action)
                        for prev_action, querier in zip(prev_actions, self.queriers)
                    ]
                ).sum(dim=0)
            )
            logits = self.pointer(encoded_input, query, mask, c=c, temp=temp)

            probs = distrib.Categorical(logits=logits)
            idx = probs.sample()

            idx_list.append(idx)  # Tour index
            log_probs.append(probs.log_prob(idx))
            entropies.append(probs.entropy())
            mask = mask + torch.zeros([batch_size, seq_len], device=inputs.device).scatter_(1, idx.unsqueeze(1), 1)

            action_rep = inputs[torch.arange(batch_size), idx]
            prev_actions.pop(0)
            prev_actions.append(action_rep)

        idx_list.append(idx_list[0])  # Return to start
        tour = torch.stack(idx_list, dim=1)  # Permutations
        log_probs = sum(log_probs)  # log-probs for backprop (Reinforce)
        entropies = sum(entropies)

        return tour, log_probs, entropies


class Critic(nn.Module):
    def __init__(self, n_hidden=128, att_hidden=256, crit_hidden=256):
        """ Critic module, estimating the minimum length of the tour from the
        encoded inputs.

        Args:
            n_hidden (int, optional): Size of the encoded input. Defaults to 128.
            att_hidden (int, optional): Attention hidden size. Defaults to 256.
            crit_hidden (int, optional): Critic hidden size. Defaults to 256.
        """
        super().__init__()
        self.glimpse = FullGlimpse(n_hidden, att_hidden)
        self.hidden = nn.Linear(n_hidden, crit_hidden)
        self.output = nn.Linear(crit_hidden, 1)

    def forward(self, inputs):
        frame = self.glimpse(inputs)
        hidden_out = F.relu(self.hidden(frame))
        preds = self.output(hidden_out).squeeze()
        return preds
