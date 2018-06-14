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

        enc_output_size = self.encoder.rnn.hidden_size
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

        enc_final, memory_bank = self.encoder(src, lengths)
        # enc_final is (layers*directions) x batch x dim.
        # Throw in the second modality for each batch sample
        # => (layers*directions) x batch x (dim + secondDim)

        batch_size = enc_final[0].size(1)
        second_modality = self.second_encoder(second_src)

        decoder_input = [None, None]
        for i, enc in enumerate(enc_final):
            padded_final = self.enc_pad_layer(enc)
            padded_second = self.second_pad_layer(second_modality)

            concatenated = padded_final + padded_second.expand_as(padded_final)
            decoder_input[i] = self.merge_layer(concatenated)
        decoder_input = tuple(decoder_input)

        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, decoder_input)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        return decoder_outputs, attns, dec_state
