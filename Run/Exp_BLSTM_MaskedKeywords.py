import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy
from Loader_NCLS import build_mask_dataset
from ModelStructure.BiLSTMMaskedKeywordsModel import BiLSTMMaskedKeywordsModel

if __name__ == '__main__':
    save_path = "E:/ProjectData/NCLS/MaskedKeywordsModel-BiLSTM/"
    fields, train_dataset = build_mask_dataset(use_part='train', word_flag=False, sample_number=100)

    # _, valid_dataset = build_mask_dataset(use_part='valid', word_flag=False)
    # _, test_dataset = build_mask_dataset(use_part='test', word_flag=False)

    model = BiLSTMMaskedKeywordsModel(vocab_sizes=len(fields.vocab), pad_ids=fields.pad_id)

    for batch_data in train_dataset:
        model(torch.LongTensor(batch_data.src).unsqueeze(0), torch.LongTensor(batch_data.tgt).unsqueeze(0))
        exit()
