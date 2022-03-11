import os
import jieba
import tqdm
import torch
import random
from typing import List
from typing import Dict
from Tools import get_device
from collections import namedtuple

EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
load_path = 'C:/PythonProject/Study202203Restart/Pretreatment/'
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


class TranslationDataset(object):
    def __init__(self, treat_data, batch_size: int, device: torch.device, train: bool, fields: Dict[str, Field]):
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.tgt))

        examples = []
        for treat_sample in tqdm.tqdm(treat_data):
            treat_article = treat_sample['Article'].lower().strip().split()[0:2048]
            treat_label = jieba.lcut(treat_sample['CrossLingualSummary'])
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


def build_dataset(sample_number=None, use_part='train', batch_shape_limit=1024):
    src_field = Field(unk=True, pad=True, bos=False, eos=False)
    tgt_field = Field(unk=True, pad=True, bos=True, eos=True)

    with open(load_path + 'SharedDictionary.vocab', 'r', encoding='UTF-8')as file:
        dictionary_words = [line.strip() for line in file]
    src_field.load_vocab(dictionary_words, tgt_field.special)
    tgt_field.load_vocab(dictionary_words, tgt_field.special)
    treat_data = ncls_loader(sample_number, use_part)

    return TranslationDataset(treat_data, batch_shape_limit, device, False, {'src': src_field, 'tgt': tgt_field})


if __name__ == '__main__':
    import numpy

    train_dataset = build_dataset(use_part='train')
    counter = 0
    for sample in train_dataset:
        counter += 1
        if counter < 38295: continue
        print(counter, numpy.shape(sample.src), numpy.shape(sample.tgt))
        if counter > 38295 + 10: break
        # exit()
    # print('\n', len(result))
    # print(result[2357])
    exit()
    for sample in result[0]:
        print(sample, result[0][sample])