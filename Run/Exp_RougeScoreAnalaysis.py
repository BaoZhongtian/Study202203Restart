import os
import json
import numpy
from rouge_score import rouge_scorer
from Loader_NCLS import build_dataset, ncls_loader
import datasets

if __name__ == '__main__':
    metrics = datasets.load_metric('rouge')
    print('HERE')
    load_path = 'E:/ProjectData/NCLS/Predict-Another-Character/'

    valid_dataset = build_dataset(use_part='test', batch_shape_limit=256, word_flag=False)

    fields = valid_dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    total_predict, total_label = [], []
    for filename in os.listdir(load_path):
        treat_data = json.load(open(load_path + filename, 'r'))
        # for sample in treat_data:
        #     print(sample)
        total_predict.extend(treat_data['hypothesis'])
        total_label.extend(treat_data['references'])
    # total_label = total_label[1:]
    print(len(total_label), len(total_predict))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    total_score = []
    for index in range(len(total_predict)):
        # print(valid_dataset.fields["tgt"].decode(total_label[index]).replace(' ', ''))
        # print(valid_dataset.fields["tgt"].decode(total_predict[index]).replace(' ', ''))
        # exit()
        # metrics.add_batch(
        #     predictions=[
        #         valid_dataset.fields["tgt"].decode(total_predict[index]).replace(' ', '').replace('<unk>', '')],
        #     references=[valid_dataset.fields["tgt"].decode(total_label[index]).replace(' ', '').replace('<unk>', '')])
        result = metrics.add_batch(references=[total_label[index]], predictions=[total_predict[index]])
        # total_score.append([result['rouge1'].fmeasure, result['rouge2'].fmeasure, result['rougeL'].fmeasure])
        # print(total_score)

    result = metrics.compute()
    for sample in result:
        print(sample, result[sample].mid)
    # print(numpy.average(total_score, axis=0))
