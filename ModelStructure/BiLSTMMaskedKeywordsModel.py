import torch
import numpy

from beaver.model.embeddings import Embedding
from beaver.model.nmt_model import Generator
from beaver.loss.optimizers import LabelSmoothingLoss


class BiLSTMMaskedKeywordsModel(torch.nn.Module):
    def __init__(self, vocab_sizes, pad_ids):
        super(BiLSTMMaskedKeywordsModel, self).__init__()
        self.embedding_layer = Embedding(
            embedding_dim=512, dropout=0.2, padding_idx=pad_ids, vocab_size=vocab_sizes)
        self.BiLSTM_Layer = torch.nn.LSTM(input_size=512, hidden_size=512, dropout=0.1, num_layers=2)
        self.generator = Generator(hidden_size=512, tgt_vocab_size=vocab_sizes)
        self.loss = LabelSmoothingLoss(label_smoothing=0, tgt_vocab_size=vocab_sizes, ignore_index=pad_ids)

    def forward(self, input_data, label=None):
        embedding_result = self.embedding_layer(input_data)
        bi_lstm_result, bi_lstm_state = self.BiLSTM_Layer(embedding_result)
        predict = self.generator(bi_lstm_result)

        if label is not None:
            loss = self.loss(predict[:, :-1, :], label)
            return loss
        return predict
