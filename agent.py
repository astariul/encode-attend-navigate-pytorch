import torch
import torch.nn as nn

from module import Embedding, Encoder, Decoder, Critic


class Agent(nn.Module):
    def __init__(self, space_dim=2, embed_hidden=128, enc_stacks=3, ff_hidden=512, enc_heads=16, query_hidden=360, att_hidden=256, crit_hidden=256, n_history=3, p_dropout=0.1):
        """ Agent, made of an encoder + decoder for the actor part, and a critic
        part.

        Args:
            input_embed (int, optional): Size of the encoded input. Defaults to 128.
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