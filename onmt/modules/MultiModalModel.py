import torch

import torch.nn as nn

from onmt.Models import EncoderBase, RNNDecoderBase


class MultiModalModel(nn.Module):

    def __init__(self, encoder: EncoderBase, second_encoder: nn.Module,
                 second_dim: int, decoder: RNNDecoderBase, generator):
        """
        Create new MultiModalModel.

        :param encoder: text encoder to use
        :param second_encoder: second modality encoder to use. Its output should be a tensor of
        size [batch x second_dim]
        :param second_dim: output dimension of second_encoder
        :param decoder: decoder to use
        :param generator: generator to use
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.second_encoder = second_encoder
        self.second_dim = second_dim

        directions = 2 if self.decoder.bidirectional_encoder else 1
        enc_output_size = self.encoder.rnn.hidden_size * directions
        self.enc_pad_layer = nn.ConstantPad1d((0, self.second_dim), 0)
        self.second_pad_layer = nn.ConstantPad1d((enc_output_size, 0), 0)
        self.merge_layer = nn.Linear(enc_output_size + self.second_dim,
                                     self.decoder.hidden_size, bias=True)

    def forward(self, src, second_src, tgt, lengths, dec_state=None):
        """
        Forward.

        :param src: the src text tensor as expected by encoder
        :param second_src: the second src tensor as expected by the second_encoder
        :param tgt: the tgt text tensor as expected by decoder
        :param lengths: src lengths pre-padding as expected by encoder
        :param dec_state: initial decoder state as expected by decoder
        :return: see NMTModel
        """

        tgt = tgt[:-1]  # exclude last target from inputs

        _, memory_bank, enc_state = self.run_encoder_to_decoder_state(src, second_src, lengths)

        # tgt 16, 16, 1
        # memory_bank 21, 16, 500
        # lengths 16
        # enc_state ([2, 16, 500], [2, 16, 500])

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        return decoder_outputs, attns, dec_state

    def run_encoder_to_decoder_state(self, src, second_src, lengths):
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_final = tuple([self._fix_enc_hidden(h) for h in enc_final])

        # enc_final is now layers x batch x enc_final_dim.
        # Throw in the second modality for each batch sample
        # => layers x batch x (dim + secondDim)
        # apply merge layer to reduce to dim expected by decoder

        second_modality = self.second_encoder(second_src)

        decoder_input = [None, None]
        for i, enc in enumerate(enc_final):
            padded_final = self.enc_pad_layer(enc)
            padded_second = self.second_pad_layer(second_modality)

            concatenated = padded_final + padded_second.expand_as(padded_final)
            decoder_input[i] = self.merge_layer(concatenated)
        decoder_input = tuple([self._refix_dec_init(h) for h in decoder_input])

        dec_state = \
            self.decoder.init_decoder_state(src, memory_bank, decoder_input)
        return enc_final, memory_bank, dec_state

    def _fix_enc_hidden(self, h):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        if self.decoder.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _refix_dec_init(self, h):
        # The decoder expects its init_decoder_state input to be (layers*directions) x batch x dim
        # We already fixed it above (to layers x batch x directions*dim), so need to revert it now
        if self.decoder.bidirectional_encoder:
            dim = h.size(2) // 2
            h_even, h_odd = torch.split(h, dim, 2)
            h_in_order = [x[i] for i in range(h_even.size(0)) for x in (h_even, h_odd)]
            h = torch.cat([x.unsqueeze(0) for x in h_in_order], dim=0)
        return h