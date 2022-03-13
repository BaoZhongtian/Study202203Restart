import os
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from ModelStructure.option import option
import torch
from Tools import get_device, ProgressBar
from Loader_NCLS import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, printing_opt

device = get_device()


def valid(model, criterion, valid_dataset, step, saver):
    model.eval()
    total_loss, total = 0.0, 0

    hypothesis, references = [], []

    for batch in tqdm.tqdm(valid_dataset):
        scores = model(batch.src, batch.tgt)
        loss = criterion(scores, batch.tgt)
        total_loss += loss.data
        total += 1

        if opt.tf:
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis += [valid_dataset.fields["tgt"].decode(p) for p in predictions]
        references += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]

    bleu = calculate_bleu(hypothesis, references)
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, bleu, total_loss / total)


def train(model, criterion, optimizer, train_dataset, valid_dataset, saver):
    total_loss = 0.0
    step_counter = 0
    model.zero_grad()

    pbar = ProgressBar(n_total=20 * len(train_dataset))
    for epoch in range(20):
        for i, batch in enumerate(train_dataset):
            step_counter += 1
            # if step_counter < 35000: continue

            scores = model(batch.src, batch.tgt)
            loss = criterion(scores, batch.tgt)
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
    opt.word_flag = False
    opt.model_path = "E:/ProjectData/NCLS/Beaver-AllData-Character"

    train_dataset = build_dataset(use_part='train', word_flag=opt.word_flag)
    valid_dataset = build_dataset(use_part='valid', word_flag=opt.word_flag)
    fields = valid_dataset.fields = train_dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    # checkpoint = torch.load("E:/ProjectData/NCLS/Beaver-AllData-220307-121224/checkpoint-step-035000")
    # model.load_state_dict(checkpoint["model"])

    saver = Saver(opt)
    criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), opt.lr)
    train(model, criterion, optimizer, train_dataset, valid_dataset, saver)
