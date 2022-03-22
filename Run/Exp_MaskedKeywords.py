import os
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ModelStructure.option import option
import torch
import numpy
from Tools import get_device, ProgressBar
from Loader_NCLS import build_mask_dataset
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.loss.optimizers import LabelChooseLoss
from beaver.model import NMTModel, MaskedKeywordsModel
from beaver.utils import Saver

device = get_device()


def valid(model, criterion, valid_dataset, step, saver):
    model.eval()
    total_loss, total = 0.0, 0

    for batch in tqdm.tqdm(valid_dataset):
        src = torch.LongTensor(batch.src).unsqueeze(0).to(device)
        tgt = torch.LongTensor(batch.tgt).unsqueeze(0).to(device)
        scores = model(src, tgt)
        loss = criterion(scores, tgt)
        total_loss += loss.data
        total += 1

    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, -1.0, total_loss / total)


def train(model, criterion, optimizer, train_dataset, valid_dataset, saver):
    total_loss = 0.0
    step_counter = 0
    model.zero_grad()

    pbar = ProgressBar(n_total=20 * len(train_dataset))
    for epoch in range(20):
        for i, batch in enumerate(train_dataset):
            step_counter += 1
            src = torch.LongTensor(batch.src).unsqueeze(0).to(device)
            tgt = torch.LongTensor(batch.tgt).unsqueeze(0).to(device)
            # if step_counter < 35000: continue

            scores = model(src, tgt)
            loss = criterion(scores, tgt)
            loss.backward()
            total_loss += loss.data

            if (i + 1) % opt.grad_accum == 0:
                optimizer.step()
                model.zero_grad()

                # print("\rstep: %7d\t loss: %7f" % (step_counter, loss.data), end='')
                pbar(epoch * len(train_dataset) + i, {'loss': loss.data})
                if step_counter % opt.report_every == 0:
                    print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                    total_loss = 0.0
                    with torch.set_grad_enabled(False):
                        valid(model, criterion, valid_dataset, step_counter, saver)
                    # checkpoint = {"model": model.state_dict(), "opt": opt}
                    # saver.save(checkpoint, step_counter, 0.0, total_loss)

                # if optimizer.n_step % opt.save_every == 0:
                #     with torch.set_grad_enabled(False):
                #         valid(model, criterion, valid_dataset, optimizer.n_step)
                #     model.train()
            del loss


if __name__ == '__main__':
    opt = option()
    opt.word_flag = True
    opt.model_path = "E:/ProjectData/NCLS/MaskedKeywordsModel-Test"
    fields, train_dataset = build_mask_dataset(use_part='train', word_flag=False)
    print(len(train_dataset))
    exit()
    test_dataset = train_dataset
    # _, valid_dataset = build_mask_dataset(use_part='valid', word_flag=False)
    # _, test_dataset = build_mask_dataset(use_part='test', word_flag=False)

    pad_ids = {"src": fields.pad_id, "tgt": fields.pad_id}
    vocab_sizes = {"src": len(fields.vocab), "tgt": len(fields.vocab)}
    model = MaskedKeywordsModel.load_model(opt, pad_ids, vocab_sizes).to(device)

    saver = Saver(opt)
    # criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)
    criterion = LabelChooseLoss(vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), opt.lr)
    train(model, criterion, optimizer, train_dataset, test_dataset, saver)
