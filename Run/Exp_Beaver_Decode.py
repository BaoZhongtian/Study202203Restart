import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    valid_dataset = build_dataset(use_part='test', batch_shape_limit=256, word_flag=False)
    fields = valid_dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    opt = option()
    opt.beam_size = 1
    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    checkpoint = torch.load("E:/ProjectData/NCLS/Beaver-AllData-Character-220312-125400/checkpoint-step-2500000")
    model.load_state_dict(checkpoint["model"])

    model.eval()
    counter = 0
    save_path = 'E:/ProjectData/NCLS/Predict-Another-Character-2500000/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    for index, batch in enumerate(tqdm.tqdm(valid_dataset)):
        if os.path.exists(save_path + '%08d.json' % index): continue
        with open(save_path + '%08d.json' % index, 'w'):
            pass
        # print(numpy.shape(batch.src))
        predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis = [p.detach().cpu().numpy().tolist() for p in predictions]
        references = [t.detach().cpu().numpy().tolist() for t in batch.tgt]

        json.dump({'hypothesis': hypothesis, 'references': references}, open(save_path + '%08d.json' % index, 'w'))
