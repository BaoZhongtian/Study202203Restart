import os
import json
import numpy
import tqdm
from rouge_score import rouge_scorer
from Loader_NCLS import build_dataset, ncls_loader
import datasets
from Loader_WikiLingual import build_wiki_lingual
from transformers import MT5Tokenizer

if __name__ == '__main__':
    metrics = datasets.load_metric('rouge')
    load_path = 'D:/ProjectData/WikiLingualResult-ZH2EN/'
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')

    total_predict = []
    for filename in os.listdir(load_path):
        with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
            sentence = file.read().replace('<pad>', '').replace('</s>', '')
        total_predict.append(sentence)
    print(len(total_predict))

    # test_data = ncls_loader(use_part='test')
    train_data, test_data = build_wiki_lingual()
    total_label = [_['EnglishSummary'] for _ in test_data]

    print(total_label[0])
    print(total_predict[0])
    # exit()

    total_predict = tokenizer.batch_encode_plus(total_predict, max_length=512, add_special_tokens=False)['input_ids']
    total_label = tokenizer.batch_encode_plus(total_label, max_length=512, add_special_tokens=False)['input_ids']

    total_score = []
    for index in tqdm.trange(len(total_predict)):
        # print(valid_dataset.fields["tgt"].decode(total_label[index]).replace(' ', ''))
        # print(valid_dataset.fields["tgt"].decode(total_predict[index]).replace(' ', ''))
        # exit()
        # metrics.add_batch(
        #     predictions=[
        #         valid_dataset.fields["tgt"].decode(total_predict[index]).replace(' ', '').replace('<unk>', '')],
        #     references=[valid_dataset.fields["tgt"].decode(total_label[index]).replace(' ', '').replace('<unk>', '')])
        result = metrics.add_batch(references=[[_ for _ in total_label[index]]],
                                   predictions=[[_ for _ in total_predict[index]]])
        # total_score.append([result['rouge1'].fmeasure, result['rouge2'].fmeasure, result['rougeL'].fmeasure])
        # print(total_score)

    result = metrics.compute()
    for sample in result:
        print(sample, result[sample].mid)
    # print(numpy.average(total_score, axis=0))
