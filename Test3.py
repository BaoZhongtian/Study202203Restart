import os
import json
import numpy
import tqdm
from rouge_score import rouge_scorer
import datasets
from transformers import MT5Tokenizer

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    metrics = datasets.load_metric('rouge')
    print('HERE')
    load_path = 'D:/ProjectData/WikiLingualResult2/WikiLingualResult-MultiTask-English2Portuguese/'

    total_predict, total_label = [], []
    for filename in tqdm.tqdm(os.listdir(load_path)):
        if os.path.isdir(os.path.join(load_path, filename)): continue

        with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
            raw_data = file.readlines()

            for sample in raw_data:
                print(sample)
            exit()
        if len(raw_data) != 2: continue

        total_predict.append(raw_data[0].lower())
        total_label.append(raw_data[1].lower())
    # total_label = total_label[1:]
    print(len(total_label), len(total_predict))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    total_score = []
    for index in tqdm.trange(len(total_predict)):
        # print(tokenizer.encode(total_predict[index], add_special_tokens=False))
        # print(tokenizer.encode(total_label[index], add_special_tokens=False))
        # exit()
        result = metrics.add_batch(references=total_label[index:index + 1], predictions=total_predict[index:index + 1])
        # result = metrics.add_batch(
        #     references=tokenizer.batch_encode_plus(total_label[index:index + 1], add_special_tokens=False)['input_ids'],
        #     predictions=tokenizer.batch_encode_plus(total_predict[index:index + 1], add_special_tokens=False)[
        #         'input_ids'])

    result = metrics.compute()
    for sample in result:
        print(sample, result[sample].mid)
    # print(numpy.average(total_score, axis=0))
