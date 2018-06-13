from onmt.io.DatasetBase import ONMTDatasetBase


class MultiModalDataset(ONMTDatasetBase):

    def __init__(self, fields, src_examples_iter,
                 src2_examples_iter, second_data_type,
                 tgt_examples_iter,
                 num_src_feats=0, num_src2_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 # TODO support for dynamic dict
                 use_filter_pred=True):
        self.data_type = 'multi'
        self.first_data_type = 'text'
        self.second_data_type = second_data_type

        self.n_src_feats = num_src_feats
        self.n_sr2c_feats = num_src2_feats
        self.n_tgt_feats = num_tgt_feats

        examples_iter = (self._join_dicts(src, src2, tgt) for src, src2, tgt in
                         zip(src_examples_iter, src2_examples_iter, tgt_examples_iter))

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        src_size = 0
        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                   and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(MultiModalDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """Sort using src and tgt sentence length (only using first modality)
        TODO implement using second modality here
        """
        return len(ex.src), len(ex.tgt)
