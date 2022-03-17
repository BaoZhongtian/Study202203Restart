import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy
import tqdm
from Tools import ProgressBar
from Loader_NCLS import build_mask_dataset
from ModelStructure.option import option
from ModelStructure.BiLSTMMaskedKeywordsModel import BiLSTMMaskedKeywordsModel
from beaver.utils import Saver

if __name__ == '__main__':
    opt = option()
    opt.word_flag = False
    opt.model_path = "E:/ProjectData/NCLS/MaskedKeywordsModel-BiLSTM-Character"
    saver = Saver(opt)

    fields, train_dataset = build_mask_dataset(use_part='train', word_flag=opt.word_flag)
    # _, valid_dataset = build_mask_dataset(use_part='valid', word_flag=opt.word_flag)
    _, test_dataset = build_mask_dataset(use_part='test', word_flag=opt.word_flag)

    model = BiLSTMMaskedKeywordsModel(vocab_sizes=len(fields.vocab), pad_ids=fields.pad_id)
    model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), 1E-4)

    pbar = ProgressBar(n_total=20 * len(train_dataset))
    step_counter = 0
    total_loss = 0.0

    for epoch in range(20):
        for batch_data in train_dataset:
            step_counter += 1
            input_data = torch.LongTensor(batch_data.src).unsqueeze(0).cuda()
            label = torch.LongTensor(batch_data.tgt).unsqueeze(0).cuda()
            loss = model(input_data, label)

            loss.backward()
            total_loss += loss.data

            optimizer.step()
            model.zero_grad()

            # print("\rstep: %7d\t loss: %7f" % (step_counter, loss.data), end='')
            pbar(step_counter, {'loss': loss.data})
            if step_counter % 5000 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                with torch.set_grad_enabled(False):
                    for batch in tqdm.tqdm(test_dataset):
                        input_data = torch.LongTensor(batch_data.src).unsqueeze(0).cuda()
                        label = torch.LongTensor(batch_data.tgt).unsqueeze(0).cuda()
                        loss = model(input_data, label)
                        total_loss += loss.data

                checkpoint = {"model": model.state_dict()}
                saver.save(checkpoint, step_counter, -1.0, total_loss)
                total_loss = 0.0
            # exit()
