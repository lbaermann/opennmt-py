import torch

import torch.nn as nn

from onmt.Models import EncoderBase, RNNDecoderBase, NMTModel


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

    def forward(self, src, second_src, tgt, lengths, dec_state=None):
        """
        Run a forward pass on the MultiModalModel. This takes the same arguments as NMTModel,
        but additionally requires a second_src tensor as expected by this MMM's second_encoder.

        :param src: the src text tensor as expected by encoder
        :param second_src: the second src tensor as expected by the second_encoder
        :param tgt: the tgt text tensor as expected by decoder
        :param lengths: src lengths pre-padding as expected by encoder
        :param dec_state: initial decoder state as expected by decoder
        :return: see NMTModel
        """

        tgt = tgt[:-1]  # exclude last target from inputs

        _, memory_bank, enc_state = self.run_encoder_to_decoder_state(src, second_src, lengths)

        decoder_outputs, dec_state, attns = \
            self.run_decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             lengths, second_src)
        return decoder_outputs, attns, dec_state

    def run_encoder_to_decoder_state(self, src, second_src, lengths):
        """
        Forward the given src and second_src up to the initial decoder state.
        :param src: the src tensor
        :param second_src: the second_src tensor
        :param lengths: the src lengths
        :return: (enc_final, memory_bank, dec_state) triple, containing the final encoder state,
        the final encoder memory bank and the initial state to use for the decoder.
        """
        raise NotImplementedError

    def run_decoder(self, tgt, memory_bank, dec_init, memory_lengths, second_src):
        return self.decoder(tgt, memory_bank, dec_init,
                            memory_lengths=memory_lengths)


class HiddenStateMergeLayerMMM(MultiModalModel):
    """
    Implementation of MultiModalModel merging the primary and secondary sources at their
    hidden representations between the encoder and decoder.
    Therefore, this model uses linear layer which gets the final encoder state and the
    second_encoder output and generates the initial decoder state to use.
    """
    def __init__(self, encoder: EncoderBase, second_encoder: nn.Module,
                 second_dim: int, decoder: RNNDecoderBase, generator):
        super().__init__(encoder, second_encoder, second_dim, decoder, generator)

        directions = 2 if self.decoder.bidirectional_encoder else 1
        enc_output_size = self.encoder.rnn.hidden_size * directions
        self.enc_pad_layer = nn.ConstantPad1d((0, self.second_dim), 0)
        self.second_pad_layer = nn.ConstantPad1d((enc_output_size, 0), 0)
        self.merge_layer = nn.Linear(enc_output_size + self.second_dim,
                                     self.decoder.hidden_size, bias=True)

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


class FirstViewThenListenMMM(MultiModalModel):
    """
    Implementation of MultiModalModel using the (encoded) second modality as initial
    cell state for the RNNEncoder. Thereby, the second_encoder returns a Tensor of size
    [batch, second_dim] while the rnn needs a [num_layers * num_directions, batch, hidden_size]
    cell state, so a linear layer changing the last dimension from second_dim to hidden_size is used
    while the first dimension of size (num_layers * num_directions) is simply expanded.
    """

    def __init__(self, encoder: EncoderBase, second_encoder: nn.Module,
                 second_dim: int, decoder: RNNDecoderBase, generator):
        super().__init__(encoder, second_encoder, second_dim, decoder, generator)

        self.convert_to_enc_init_layer = nn.Linear(second_dim, self.encoder.rnn.hidden_size)

    def run_encoder_to_decoder_state(self, src, second_src, lengths):
        second_encoded = self.second_encoder(second_src)  # [batch x second_dim]
        converted: torch.Tensor = self.convert_to_enc_init_layer(second_encoded)  # [batch x hidden_size]
        converted = converted.unsqueeze(0)  # [1 x batch x hidden_size]
        first_dim = self.encoder.rnn.num_layers * (2 if self.encoder.rnn.bidirectional else 1)
        encoder_init = converted.expand(first_dim, -1, -1)  # [first_dim x batch x hidden_size]

        if isinstance(self.encoder.rnn, nn.LSTM):
            encoder_init = (encoder_init, encoder_init)  # Use it as initial hidden and cell state

        enc_final, memory_bank = self.encoder(src, lengths, encoder_state=encoder_init)
        dec_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        return enc_final, memory_bank, dec_state


class GeneratorMergeMMM(MultiModalModel):
    """
    Implementation of MultiModalModel using the (encoded) second modality solely as additional
    input to the generator. This means, the output of the second encoder is simply concatenated
    to the output of the decoder.
    Note: A fitting generator must be used with this model so that dimensions match!
    """

    def __init__(self, encoder: EncoderBase, second_encoder: nn.Module,
                 second_dim: int, decoder: RNNDecoderBase, generator):
        super().__init__(encoder, second_encoder, second_dim, decoder, generator)

    def run_encoder_to_decoder_state(self, src, second_src, lengths):
        return NMTModel.run_encoder_to_decoder_state(self, src, lengths)

    def run_decoder(self, tgt, memory_bank, dec_init, memory_lengths, second_src):
        second_encoded: torch.Tensor = self.second_encoder(second_src)
        out, state, attn = super().run_decoder(tgt, memory_bank, dec_init,
                                               memory_lengths, second_src)

        # decoder output is [len x batch x rnn_size]
        # second encoded is [batch x second_dim]
        # concat it to [len x batch x (rnn_size + second_dim)]

        length = out.size(0)
        unsqueezed = second_encoded.unsqueeze(0)  # [1 x batch x rnn_size]
        second_expanded = unsqueezed.expand(length, -1, -1)  # [len x batch x rnn_size]
        concat = torch.cat((out, second_expanded), dim=2)
        return concat, state, attn
