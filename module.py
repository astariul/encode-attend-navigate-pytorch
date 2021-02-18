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


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden=512, num_heads=16, p_dropout=0.1):
        """ Module applying a single block of multi-head attention.

        Args:
            n_hidden (int, optional): Hidden size. Should be a multiple of
                `num_heads`. Defaults to 126.
            num_heads (int, optional): Number of heads. Defaults to 16.
            p_dropout (float, optional): Dropout rate. Defaults to 0.1.

        Raises:
            ValueError: Error raised if `n_hidden` is not a multiple of
                `num_heads`.
        """
        super().__init__()
        if n_hidden % num_heads != 0:
            raise ValueError(
                "`n_hidden` ({}) should be a multiple of `num_heads` ({})".format(
                    n_hidden, num_heads
                )
            )

        self.q = nn.Linear(n_hidden, n_hidden)
        self.k = nn.Linear(n_hidden, n_hidden)
        self.v = nn.Linear(n_hidden, n_hidden)

        self.dropout = nn.Dropout(p_dropout)
        self.batch_norm = nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01)
        # (Use TF default parameter, for consistency with original code)

        self.num_heads = num_heads

    def forward(self, inputs):
        # inputs : [bs, seq, n_hidden]
        bs, _, n_hidden = inputs.size()

        # Linear projections
        q = F.relu(self.q(inputs))
        k = F.relu(self.k(inputs))
        v = F.relu(self.v(inputs))

        # Split and concat
        q_ = torch.cat(torch.split(q, n_hidden // self.num_heads, dim=2))
        k_ = torch.cat(torch.split(k, n_hidden // self.num_heads, dim=2))
        v_ = torch.cat(torch.split(v, n_hidden // self.num_heads, dim=2))

        # Multiplication
        outputs = torch.matmul(q_, torch.transpose(k_, 2, 1))

        # Scale
        outputs = outputs / (k_.size(-1) ** 0.5)

        # Activation
        outputs = F.softmax(outputs, dim=-1)

        # Dropout
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, v_)

        # Restore shape
        outputs = torch.cat(torch.split(outputs, bs), dim=2)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = self.batch_norm(outputs.transpose(1, 2)).transpose(1, 2)

        return outputs


class FeedForward(nn.Module):
    def __init__(self, layers_size=[512, 2048, 512]):
        """ Feed Forward network.

        Args:
            layers_size (list, optional): List describing the internal sizes of
                the FF network. Defaults to [512, 2048, 512].
        """
        super().__init__()

        self.layers = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            self.layers.append(nn.Linear(in_size, out_size))

        self.batch_norm = nn.BatchNorm1d(layers_size[-1], eps=0.001, momentum=0.01)
        # (Use TF default parameter, for consistency with original code)

    def forward(self, inputs):
        # inputs : [bs, seq, n_hidden]
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)

            if i < len(self.layers) - 1:
                outputs = F.relu(outputs)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = self.batch_norm(outputs.tranpose(1, 2)).tranpose(1, 2)

        return outputs


class Encoder(nn.module):
    def __init__(
        self, num_layers=3, ff_hidden=2048, n_hidden=512, num_heads=16, p_dropout=0.1
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
        self.num_layers = num_layers
        self.multihead_attention = MultiHeadAttention(n_hidden, num_heads, p_dropout)
        self.ff = FeedForward(layers_size=[n_hidden, ff_hidden, n_hidden])

    def forward(self, input_seq):
        for _ in range(self.num_layers):
            input_seq = self.multihead_attention(input_seq)
            input_seq = self.ff(input_seq)
        return input_seq


class Pointer(nn.Module):
    def __init__(self, query_dim=256, n_hidden=512):
        """ Pointer network.

        Args:
            query_dim (int, optional): Dimension of the query. Defaults to 256.
            n_hidden (int, optional): Hidden size. Defaults to 512.
        """
        super().__init__()
        self.w_q = nn.Linear(query_dim, n_hidden, bias=False)
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_hidden)))

    def forward(self, encoded_ref, query, mask, c=10, temp=1):
        encoded_query = self.w_q(query).unsqueeze(1)
        scores = torch.sum(self.v * F.tanh(encoded_ref + encoded_query), dim=-1)
        scores = c * F.tanh(scores / temp)
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
        self.conv = nn.Conv1d(in_dim, out_dim, 1)
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(out_dim)))

    def forward(self, ref):
        # Attention
        encoded_ref = self.conv(ref)
        scores = torch.sum(self.v * F.tanh(encoded_ref), dim=-1)
        attention = F.softmax(scores)

        # Glimpse : Linear combination of reference vectors (define new query vector)
        glimpse = torch.matmul(ref, attention.unsqueeze(-1))
        glimpse = torch.sum(glimpse, dim=1)
        return glimpse


class Decoder(nn.Module):
    def __init__(self, n_hidden=512, dec_hidden=256, query_dim=360, n_history=3):
        """ Decoder with a Pointer network and a memory of size `n_history`. 

        Args:
            n_hidden (int, optional): Encoder hidden size. Defaults to 512.
            dec_hidden (int, optional): Decoder hidden size. Defaults to 256.
            query_dim (int, optional): Dimension of the query. Defaults to 360.
            n_history (int, optional): Size of history. Defaults to 3.
        """
        self.conv = nn.Conv1d(n_hidden, dec_hidden, 1)
        self.n_history = n_history
        self.queriers = [
            nn.Linear(n_hidden, query_dim, bias=False) for _ in range(n_history)
        ]
        self.pointer = Pointer(query_dim, n_hidden)

    def forward(self, inputs, c=10, temp=1):
        batch_size, seq_len, hidden = inputs.size()

        idx_list, log_probs, entropies = [], [], []  # Tours index, log_probs, entropies
        mask = torch.zeros([batch_size, seq_len])  # Mask for actions

        encoded_input = self.conv(inputs)

        prev_actions = [torch.zeros([batch_size, hidden]) for _ in self.n_history]

        for _ in range(seq_len):
            query = F.ReLu(
                torch.sum(
                    [
                        querier(prev_action)
                        for prev_action, querier in zip(prev_actions, self.queriers)
                    ]
                )
            )
            logits = self.pointer(encoded_input, query, mask, c=c, temp=temp)

            probs = distrib.Categorical(logits)
            idx = probs.sample()

            idx_list.append(idx)  # Tour index
            log_probs.append(probs.log_prob(idx))
            entropies.append(probs.entropy())
            mask[idx] = 1

            action_rep = inputs[idx]
            prev_actions.pop(0)
            prev_actions.append(action_rep)

        idx_list.append(idx_list[0])  # Return to start
        tour = torch.stack(idx_list, dim=1)  # Permutations
        log_probs = torch.sum(
            log_probs
        )  # Corresponding log-probabilities for backpropagation (Reinforce)
        entropies = torch.sum(entropies)

        return tour, log_probs, entropies


class Critic(nn.Module):
    def __init__(self, input_embed=128, dec_hidden=256, crit_hidden=256):
        """ Critic module, estimating the minimum length of the tour from the
        encoded inputs.

        Args:
            input_embed (int, optional): Size of the encoded input. Defaults to 128.
            dec_hidden (int, optional): Decoder hidden size. Defaults to 256.
            crit_hidden (int, optional): Critic hidden size. Defaults to 256.
        """
        self.glimpse = FullGlimpse(input_embed, dec_hidden)
        self.hidden = nn.Linear(dec_hidden, crit_hidden)
        self.output = nn.Linear(crit_hidden, 1)

    def forward(self, inputs):
        frame = self.glimpse(inputs)
        hidden_out = F.ReLu(self.hidden(frame))
        preds = self.output(hidden_out).squeeze()
        return preds
