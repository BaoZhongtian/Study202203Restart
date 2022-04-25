import os
import numpy
import tqdm
import json
from Loader_NCLS import ncls_loader_EN2ZH

if __name__ == '__main__':
    total_keywords = []

    for use_part in ['valid', 'test', 'train']:
        train_data = ncls_loader_EN2ZH(use_part=use_part)
        for sample in tqdm.tqdm(train_data):
            article = sample['Article'].lower().strip().split()
            summary = set(sample['Summary'].lower().strip().split())
            cross_lingual_summary = [_ for _ in sample['CrossLingualSummary']]

            dictionary = {}
            for word in article:
                if word not in summary: continue
                if word in dictionary.keys():
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

            total_keywords.append(dictionary)
    json.dump(total_keywords, open('TrainKeywords.json', 'w'))
