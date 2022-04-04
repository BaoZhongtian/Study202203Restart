import os
import json
import numpy
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BartTokenizer
from Loader_NCLS import MaskedExample

load_path = 'D:/PythonProject/Study202203Restart/Pretreatment/'


class CollateClass:
    def __init__(self, tokenizer, ignore_number=50, select_keywords_number=10):
        self.tokenizer = tokenizer
        self.select_keywords_number = select_keywords_number
        self.ignore_words = set()
        with open(load_path + 'IgnoreWords.txt', 'r', encoding='UTF-8') as file:
            for _ in range(ignore_number):
                raw_text = file.readline()
                self.ignore_words.add(raw_text.split(',')[0])

    def collate(self, input_data):
        return self.collate_keywords(input_data)

    def overlap_keywords_generation(self, sample):
        treat_article = sample['article'].lower().strip().split()
        summary = set(sample['summary'].lower().strip().split())

        if len(summary) > len(treat_article): return None
        for word in self.ignore_words:
            if word in summary: summary.remove(word)

        keywords_dictionary = {}
        for word in treat_article:
            if word in summary:
                if word in keywords_dictionary.keys():
                    keywords_dictionary[word] += 1
                else:
                    keywords_dictionary[word] = 1
        keywords_tuple = [[_, keywords_dictionary[_]] for _ in keywords_dictionary]
        keywords_tuple = sorted(keywords_tuple, key=lambda x: x[-1], reverse=True)[0:self.select_keywords_number]
        treat_keywords = [_[0] for _ in keywords_tuple]
        return treat_keywords

    def collate_keywords(self, input_data):
        mask_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

        batch_summary, batch_article, batch_label = [], [], []
        for index in range(len(input_data)):
            current_summary_token = self.tokenizer.encode_plus(
                input_data[index]['summary'], add_special_tokens=True)['input_ids']

            current_lm_label = []

            #######################################
            current_keywords = self.overlap_keywords_generation(input_data[index])
            if current_keywords is None:
                print('Check')
                exit()
            current_keywords_tokens = self.tokenizer.batch_encode_plus(
                [' ' + _ for _ in current_keywords], add_special_tokens=False)['input_ids']
            current_article_token = self.tokenizer.encode_plus(
                input_data[index]['article'], add_special_tokens=True, truncation=True, max_length=512)['input_ids']

            indexX = -1
            while indexX < len(current_article_token):
                indexX += 1
                similar_flag = False
                for indexY in range(len(current_keywords_tokens)):
                    for indexZ in range(len(current_keywords_tokens[indexY])):
                        if indexX + indexZ >= len(current_article_token): break
                        if current_article_token[indexX + indexZ] != current_keywords_tokens[indexY][indexZ]: break
                    if indexX + indexZ >= len(current_article_token): continue
                    if current_article_token[indexX + indexZ] == current_keywords_tokens[indexY][indexZ]:
                        similar_flag = True
                        break
                if similar_flag:
                    for indexZ in range(len(current_keywords_tokens[indexY])):
                        current_lm_label.append(current_article_token[indexX + indexZ])
                        current_article_token[indexX + indexZ] = mask_id
                    indexX += indexZ
                    continue
                current_lm_label.append(-100)

            current_lm_label = current_lm_label[0:len(current_article_token)]
            assert len(current_lm_label) == len(current_article_token)

            batch_summary.append(current_summary_token)
            batch_article.append(current_article_token)
            batch_label.append(current_lm_label)

        padding_batch_summary, padding_batch_article, padding_batch_label = [], [], []
        treated_length = max([len(_) for _ in batch_summary])
        for index in range(len(batch_summary)):
            padding_batch_summary.append(numpy.concatenate(
                [batch_summary[index], [pad_id for _ in range(treated_length - len(batch_summary[index]))]]))

        treated_length = max([len(_) for _ in batch_article])
        for index in range(len(batch_article)):
            padding_batch_article.append(numpy.concatenate(
                [batch_article[index], [pad_id for _ in range(treated_length - len(batch_article[index]))]]))

        treated_length = max([len(_) for _ in batch_label])
        for index in range(len(batch_label)):
            padding_batch_label.append(numpy.concatenate(
                [batch_label[index], [-100 for _ in range(treated_length - len(batch_label[index]))]]))

        return MaskedExample(padding_batch_summary, padding_batch_article, padding_batch_label)


def loader_cnndm(
        batch_size=4, tokenizer=None, train_part_shuffle=True, small_data_flag=False, limit_size=None):
    if tokenizer is None: tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if small_data_flag:
        train_data = json.load(open(os.path.join(load_path, 'CNNDM_train_part_shuffle.json'), 'r'))
        test_data = json.load(open(os.path.join(load_path, 'CNNDM_test_part_shuffle.json'), 'r'))
    else:
        train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
        test_data = json.load(open(os.path.join(load_path, 'CNNDM_test.json'), 'r'))
    print('Load Completed')
    if limit_size is not None:
        train_data = train_data[0:limit_size]
        test_data = test_data[0:limit_size]

    collate = CollateClass(tokenizer)

    train_dataset = DataLoader(
        train_data, batch_size=batch_size, shuffle=train_part_shuffle, collate_fn=collate.collate)
    test_dataset = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate.collate)

    return train_dataset, test_dataset


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('D:/PythonProject/bert-base-uncased/')
    train_loader, test_loader = loader_cnndm(
        batch_size=3, tokenizer=tokenizer, small_data_flag=True, train_part_shuffle=False)
    for sample in train_loader:
        print(numpy.shape(sample.article), numpy.shape(sample.summary), numpy.shape(sample.label))
        # exit()
#
#     article_len, summary_len = [], []
#     for batch_data in tqdm.tqdm(train_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     for batch_data in tqdm.tqdm(val_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     for batch_data in tqdm.tqdm(test_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     print(article_len)
#     import json
#
#     json.dump(article_len, open('article_len.json', 'w'))
#     json.dump(summary_len, open('summary_len.json', 'w'))
