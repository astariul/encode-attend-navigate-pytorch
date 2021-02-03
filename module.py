import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv = nn.Conv1d(in_dim, out_dim, 1)
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

    def forward(self, x):
        # x : [bs, seq, in_dim]
        emb_x = self.conv(x)
        bs, seq, h = emb_x.size()
        return self.batch_norm(emb_x.view(-1, h)).view(bs, seq, h)


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
            raise ValueError("`n_hidden` ({}) should be a multiple of `num_heads` ({})".format(n_hidden, num_heads))

        self.q = nn.Linear(n_hidden, n_hidden)
        self.k = nn.Linear(n_hidden, n_hidden)
        self.v = nn.Linear(n_hidden, n_hidden)

        self.dropout = nn.Dropout(p_dropout)
        self.batch_norm = nn.BatchNorm1d(n_hidden)

        self.num_heads = num_heads

    def forward(self, inputs):
        # inputs : [bs, seq, n_hidden]

        # Linear projections
        q = F.relu(self.q(inputs))
        k = F.relu(self.k(inputs))
        v = F.relu(self.v(inputs))

        # Split and concat
        q_ = torch.cat(torch.split(q, [1, 1, self.num_heads]))
        k_ = torch.cat(torch.split(k, [1, 1, self.num_heads]))
        v_ = torch.cat(torch.split(v, [1, 1, self.num_heads]))

        # Multiplication
        outputs = torch.matmul(q_, torch.transpose(k_, 2, 1))

        # Scale
        outputs = outputs / (k_.size(-1) ** 0.5)

        # Activation
        outputs = F.softmax(outputs)

        # Dropout
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, v_)

        # Restore shape
        v_ = torch.cat(torch.split(outputs, [self.num_heads, 1, 1]), dim=2)

        # Residual connection
        outputs += inputs

        # Normalize
        bs, seq, h = outputs.size()
        outputs = self.batch_norm(outputs.view(-1, h)).view(bs, seq, h)

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
            self.layers.append(nn.Conv1d(in_size, out_size, 1))

        self.batch_norm = nn.BatchNorm1d(layers_size[-1])

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
        bs, seq, h = outputs.size()
        outputs = self.batch_norm(outputs.view(-1, h)).view(bs, seq, h)

        return outputs


class Encoder(nn.module):
    def __init__(self, num_layers=3, ff_hidden=2048, n_hidden=512, num_heads=16, p_dropout=0.1):
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