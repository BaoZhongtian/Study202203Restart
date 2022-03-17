import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy
import tqdm
import json
from ModelStructure.option import option
import torch
from Tools import get_device, ProgressBar
from Loader_NCLS import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu

device = get_device()

if __name__ == '__main__':
    valid_dataset = build_dataset(use_part='test', batch_shape_limit=1024, word_flag=False)
    fields = valid_dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    opt = option()
    criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

    opt = option()

    load_path = 'E:/ProjectData/NCLS/Beaver-AllData-Character-220312-125400/'
    save_path = 'E:/ProjectData/NCLS/Beaver-AllData-Character-220312-125400-Result/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    for filename in os.listdir(load_path):
        print(filename)
        if filename.find('checkpoint-step') == -1: continue
        if os.path.exists(os.path.join(save_path, filename + '.txt')): continue
        with open(os.path.join(save_path, filename + '.txt'), 'w'):
            pass
        model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
        checkpoint = torch.load(os.path.join(load_path, filename))
        model.load_state_dict(checkpoint["model"])

        model.eval()
        total_loss = 0.0
        for index, batch in enumerate(tqdm.tqdm(valid_dataset)):
            scores = model(batch.src, batch.tgt)
            loss = criterion(scores, batch.tgt)
            total_loss += loss.item()
            del loss
        with open(os.path.join(save_path, filename + '.txt'), 'w') as file:
            file.write(str(total_loss))
