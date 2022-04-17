from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_CNNDM import loader_cnndm
from Loader_NCLS_Neo import build_ncls_neo, build_mask_ncls_for_mt5, build_mask_ncls_treat_sample
from Tools import ProgressBar, SaveModel
import torch
import numpy
import os
import datetime
import tqdm
import json

cuda_flag = True

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/Study202203Restart/mT5_MKM_NCLS/')
    model.eval()
    train_data, val_data = build_mask_ncls_for_mt5(tokenizer=tokenizer, sample_number=1000)
    if cuda_flag: model = model.cuda()
    save_path = 'D:/ProjectData/MKM_Result/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    for i, batch in enumerate(tqdm.tqdm(val_data)):
        if batch is None: continue
        batch = build_mask_ncls_treat_sample(batch, tokenizer)
        if batch is None: continue
        input_ids, labels = batch['input_ids'], batch['labels']

        if cuda_flag:
            input_ids, labels = input_ids.cuda(), labels.cuda()
        logits = model.forward(input_ids, labels=labels).logits.argmax(dim=-1).squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        logits_sentence = tokenizer.decode(logits)
        labels_sentence = tokenizer.decode(labels)

        logits_word, labels_word = [], []
        for index in range(1, 11):
            if labels_sentence.find('<extra_id_%d>' % (index + 1)) == -1: continue

            if logits_sentence.find('<extra_id_%d>' % index) == -1 or logits_sentence.find(
                    '<extra_id_%d>' % (index + 1)) == -1:
                logits_word.append('')
            else:
                current_words = logits_sentence[logits_sentence.find('<extra_id_%d>' % index) + len(
                    '<extra_id_%d>' % index):logits_sentence.find('<extra_id_%d>' % (index + 1))]
                logits_word.append(current_words)

            current_words = labels_sentence[labels_sentence.find('<extra_id_%d>' % index) + len(
                '<extra_id_%d>' % index):labels_sentence.find('<extra_id_%d>' % (index + 1))]
            labels_word.append(current_words)

        json.dump({'logits': logits_word, 'labels': labels_word}, open(os.path.join(save_path, '%08d.json' % i), 'w'))
        # exit()
