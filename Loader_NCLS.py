import os
import json
import jieba
import tqdm
import torch
import numpy
import random
from typing import List
from typing import Dict
from Tools import get_device
from collections import namedtuple

EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
load_path = 'D:/PythonProject/Study202203Restart/Pretreatment/'
device = get_device()


def ncls_loader(sample_number=None, use_part='train'):
    with open(load_path + 'EN2ZHSUM_%s.txt' % use_part, 'r', encoding='UTF-8') as file:
        total_result, treat_sample = [], []
        treat_sentence = file.readline()

        while treat_sentence:
            treat_sample.append(treat_sentence.replace('\n', '').replace('\t', ''))
            if treat_sentence[0:6] == '</doc>':
                assert treat_sample[1] == '<Article>'
                article_sentence = treat_sample[2]
                for search_index in range(3, len(treat_sample)):
                    if treat_sample[search_index] == '</Article>': break
                    article_sentence += ' ' + treat_sample[search_index]

                summary, cross_lingual_summary = '', ''
                for index, sample in enumerate(treat_sample):
                    # print(index, sample)
                    if sample[0:7] == '<ZH-REF' and sample.find('-human-corrected>') == -1:
                        cross_lingual_summary += treat_sample[index + 1] + ' '
                    if sample[0:7] == '<EN-REF':
                        summary += treat_sample[index + 1] + ' '

                total_result.append(
                    {'Article': article_sentence, 'Summary': summary, 'CrossLingualSummary': cross_lingual_summary})
                treat_sample = []
                print('\rCurrent Load %d Samples' % len(total_result), end='')

                if sample_number is not None and len(total_result) >= sample_number: return total_result
            treat_sentence = file.readline()
    return total_result


class Field(object):
    def __init__(self, bos: bool, eos: bool, pad: bool, unk: bool):
        self.bos_token = BOS_TOKEN if bos else None
        self.eos_token = EOS_TOKEN if eos else None
        self.unk_token = UNK_TOKEN if unk else None
        self.pad_token = PAD_TOKEN if pad else None

        self.vocab = None

    def load_vocab(self, words: List[str], specials: List[str]):
        self.vocab = Vocab(words, specials)

    def process(self, batch, device):
        max_len = max(len(x) for x in batch)

        padded, length = [], []

        for x in batch:
            bos = [self.bos_token] if self.bos_token else []
            eos = [self.eos_token] if self.eos_token else []
            pad = [self.pad_token] * (max_len - len(x))

            padded.append(bos + x + eos + pad)
            length.append(len(x) + len(bos) + len(eos))

        padded = torch.tensor([self.encode(ex) for ex in padded])

        return padded.long().to(device)

    def process_with_mask(self, article, summary, keywords, device):
        bos = [self.bos_token] if self.bos_token else []
        eos = [self.eos_token] if self.eos_token else []

        article = article.copy()
        label = [self.pad_id for _ in range(len(article))]
        for index in range(len(article)):
            if article[index] in keywords:
                label[index] = self.encode([article[index]])[0]
                article[index] = '[MASK]'

        input_article = bos + summary + bos + article
        label = [self.pad_id] * len(bos + summary + bos) + label
        assert len(input_article) == len(label)

        if len(input_article) > 1024:
            input_article = input_article[0:1024]
            label = label[0:len(input_article)]

        input_article += eos
        label += [self.pad_id]

        # treat_sample = {'input': torch.LongTensor(self.encode(input_article)).unsqueeze(0).to(device),
        #                 'label': torch.LongTensor(label).unsqueeze(0).to(device)}
        return Example(self.encode(input_article), label)

    def process_with_mask_separate(self, article, summary, keywords, device):
        bos = [self.bos_token] if self.bos_token else []
        eos = [self.eos_token] if self.eos_token else []

        article = article.copy()
        label = [self.pad_id for _ in range(len(article))]
        for index in range(len(article)):
            if article[index] in keywords:
                label[index] = self.encode([article[index]])[0]
                article[index] = '[MASK]'

        assert len(article) == len(label)
        article = bos + article + eos
        label = [self.pad_id] * len(bos) + label + [self.pad_id] * len(eos)

        for index in range(len(label)):
            if label[index] == self.pad_id: label[index] = -100

        summary = bos + summary + eos
        return MaskedExample(self.encode(summary), self.encode(article), label)

    def encode(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.vocab.stoi:
                ids.append(self.vocab.stoi[tok])
            else:
                ids.append(self.unk_id)
        return ids

    def decode(self, ids):
        tokens = []
        for tok in ids:
            tok = self.vocab.itos[tok]
            if tok == self.eos_token:
                break
            if tok == self.bos_token:
                continue
            tokens.append(tok)
        # 删除BPE符号，按照T2T切分-。
        return " ".join(tokens).replace("@@ ", "").replace("@@", "").replace("-", " - ")

    @property
    def special(self):
        return [tok for tok in [self.unk_token, self.pad_token, self.bos_token, self.eos_token] if tok is not None]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]

    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def bos_id(self):
        return self.vocab.stoi[self.bos_token]

    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]


class Vocab(object):
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


Batch = namedtuple("Batch", ['src', 'tgt', 'batch_size'])
Example = namedtuple("Example", ['src', 'tgt'])
MaskedExample = namedtuple("MaskedExample", ['summary', 'article', 'label'])


class TranslationDataset(object):
    def __init__(self, treat_data, batch_size: int, device: torch.device, train: bool, fields: Dict[str, Field],
                 word_flag=True):
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))

        examples = []
        for treat_sample in tqdm.tqdm(treat_data):
            treat_article = treat_sample['Article'].lower().strip().split()[0:2048]
            if word_flag:
                treat_label = jieba.lcut(treat_sample['CrossLingualSummary'])
            else:
                treat_label = [_ for _ in treat_sample['CrossLingualSummary'].lower()]
            if len(treat_label) > len(treat_article): continue
            examples.append(Example(treat_article, treat_label))
        examples, self.seed = self.sort(examples)

        self.num_examples = len(examples)
        self.batches = list(self.batch(examples, self.batch_size))

    def __iter__(self):
        while True:
            if self.train:
                random.shuffle(self.batches)
            for minibatch in self.batches:
                if len(minibatch) == 0: continue
                src = self.fields["src"].process([x.src for x in minibatch], self.device)
                tgt = self.fields["tgt"].process([x.tgt for x in minibatch], self.device)
                yield Batch(src=src, tgt=tgt, batch_size=len(minibatch))
            if not self.train:
                break

    def __len__(self):
        return len(self.batches) - 1

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]), reverse=True)
        return sorted(examples, key=self.sort_key), seed

    @staticmethod
    def batch(data, batch_size):
        minibatch, cur_len = [], 0
        for ex in data:
            minibatch.append(ex)
            cur_len = max(cur_len, len(ex.src), len(ex.tgt))
            if cur_len * len(minibatch) > batch_size:
                yield minibatch[:-1]
                minibatch, cur_len = [ex], max(len(ex.src), len(ex.tgt))
        if minibatch:
            yield minibatch


def build_dataset(sample_number=None, use_part='train', batch_shape_limit=1024, word_flag=True):
    src_field = Field(unk=True, pad=True, bos=False, eos=False)
    tgt_field = Field(unk=True, pad=True, bos=True, eos=True)

    if word_flag:
        with open(load_path + 'SharedDictionary.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    else:
        with open(load_path + 'SharedDictionary_Character.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    src_field.load_vocab(dictionary_words, tgt_field.special)
    tgt_field.load_vocab(dictionary_words, tgt_field.special)
    treat_data = ncls_loader(sample_number, use_part)

    return TranslationDataset(treat_data, batch_shape_limit, device, False, {'src': src_field, 'tgt': tgt_field},
                              word_flag)


def build_mask_dataset(sample_number=None, use_part='train', keywords_number=10, word_flag=True, max_size=1000,
                       separate_flag=False):
    field = Field(unk=True, pad=True, bos=True, eos=True)
    if word_flag:
        with open(load_path + 'SharedDictionary.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    else:
        with open(load_path + 'SharedDictionary_Character.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    dictionary_words.append('[MASK]')

    field.load_vocab(dictionary_words, field.special)
    total_data = ncls_loader(sample_number, use_part)
    total_keywords = json.load(open(load_path + '%sKeywords.json' % use_part, 'r'))[0:len(total_data)]

    # mask_id = field.encode(['[MASK]'])[0]

    treated_samples_all = []
    for indexX in tqdm.trange(len(total_data)):
        treat_article = total_data[indexX]['Article'].lower().strip().split()[0:max_size].copy()
        if word_flag:
            treat_summary = jieba.lcut(total_data[indexX]['CrossLingualSummary'])
        else:
            treat_summary = [_ for _ in total_data[indexX]['CrossLingualSummary'].lower()]
        treat_keywords = set([_[0] for _ in total_keywords[indexX][0:keywords_number]])
        if len(treat_summary) > len(treat_article): continue

        if separate_flag:
            treated_samples_all.append(
                field.process_with_mask_separate(treat_article, treat_summary, treat_keywords, device))
        else:
            treated_samples_all.append(field.process_with_mask(treat_article, treat_summary, treat_keywords, device))
    return field, treated_samples_all


def build_overlap_mask_dataset(sample_number=None, use_part='train', keywords_number=10, ignore_number=50,
                               word_flag=True, max_size=1000, separate_flag=False):
    field = Field(unk=True, pad=True, bos=True, eos=True)
    if word_flag:
        with open(load_path + 'SharedDictionary.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    else:
        with open(load_path + 'SharedDictionary_Character.vocab', 'r', encoding='UTF-8')as file:
            dictionary_words = [line.strip() for line in file]
    dictionary_words.append('[MASK]')

    field.load_vocab(dictionary_words, field.special)
    total_data = ncls_loader(sample_number, use_part)

    ############################################

    ignore_words = set()
    with open(load_path + 'IgnoreWords.txt', 'r', encoding='UTF-8') as file:
        for _ in range(ignore_number):
            raw_text = file.readline()
            ignore_words.add(raw_text.split(',')[0])

    treated_samples_all = []
    for sample in tqdm.tqdm(total_data):
        treat_article = sample['Article'].lower().strip()[0:max_size].split()
        summary = set(sample['Summary'].lower().strip().split())
        if word_flag:
            treat_summary = jieba.lcut(sample['CrossLingualSummary'])
        else:
            treat_summary = [_ for _ in sample['CrossLingualSummary'].lower()]

        ########################################
        # Warning
        # treat_summary = sample['Summary'].lower().strip().split()
        ########################################

        if len(treat_summary) > len(treat_article): continue

        for word in ignore_words:
            if word in summary: summary.remove(word)

        keywords_dictionary = {}
        for word in treat_article:
            if word in summary:
                if word in keywords_dictionary.keys():
                    keywords_dictionary[word] += 1
                else:
                    keywords_dictionary[word] = 1
        keywords_tuple = [[_, keywords_dictionary[_]] for _ in keywords_dictionary]
        keywords_tuple = sorted(keywords_tuple, key=lambda x: x[-1], reverse=True)[0:keywords_number]
        treat_keywords = set([_[0] for _ in keywords_tuple])

        if separate_flag:
            treated_samples_all.append(
                field.process_with_mask_separate(treat_article, treat_summary, treat_keywords, device))
        else:
            treated_samples_all.append(field.process_with_mask(treat_article, treat_summary, treat_keywords, device))
    return field, treated_samples_all


if __name__ == '__main__':
    field, dataset = build_overlap_mask_dataset(word_flag=False, separate_flag=True)
    for sample in dataset:
        print("\n\n")
        print(field.decode(sample.summary))
        print(field.decode(sample.article))
        exit()
    exit()
    for sample in result[0]:
        print(sample, result[0][sample])
