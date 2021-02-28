import torch
import torch.nn as nn

from module import Embedding, Encoder, Decoder, Critic


class Agent(nn.Module):
    def __init__(self, space_dim=2, embed_hidden=128, enc_stacks=3, ff_hidden=512, enc_heads=16, query_hidden=360, att_hidden=256, crit_hidden=256, n_history=3, p_dropout=0.1):
        """Agent, made of an encoder + decoder for the actor part, and a critic
        part.

        Args:
            space_dim (int, optional): Dimension for the cities coordinates.
                Defaults to 2.
            embed_hidden (int, optional): Embeddings hidden size. Defaults to 128.
            enc_stacks (int, optional): Number of encoder layers. Defaults to 3.
            ff_hidden (int, optional): Hidden size for the FF part of the encoder.
                Defaults to 512.
            enc_heads (int, optional): Number of attention heads for the encoder.
                Defaults to 16.
            query_hidden (int, optional): Query hidden size. Defaults to 360.
            att_hidden (int, optional): Attention hidden size. Defaults to 256.
            crit_hidden (int, optional): Critic hidden size. Defaults to 256.
            n_history (int, optional): Size of history (memory size of the
                decoder). Defaults to 3.
            p_dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Actor
        self.embedding = Embedding(in_dim=space_dim, out_dim=embed_hidden)
        self.encoder = Encoder(num_layers=enc_stacks, n_hidden=embed_hidden, ff_hidden=ff_hidden, num_heads=enc_heads, p_dropout=p_dropout)
        self.decoder = Decoder(n_hidden=embed_hidden, att_dim=att_hidden, query_dim=query_hidden, n_history=n_history)

        # Critic
        self.critic = Critic(n_hidden=embed_hidden, att_hidden=att_hidden, crit_hidden=crit_hidden)

    def forward(self, inputs, c=10, temp=1):
        embed_inp = self.embedding(inputs)
        encoder_hidden = self.encoder(embed_inp)

        tour, log_probs, entropies = self.decoder(encoder_hidden)

        critique = self.critic(encoder_hidden)

        return tour, critique, log_probs, entropies