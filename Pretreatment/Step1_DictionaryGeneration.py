import jieba
import tqdm
import collections
from Loader_NCLS import ncls_loader

if __name__ == '__main__':
    counter_english = collections.Counter()
    counter_chinese = collections.Counter()

    for use_part in ['valid', 'test', 'train']:
        result = ncls_loader(use_part=use_part)
        print()
        for sample in tqdm.tqdm(result):
            counter_english.update(sample['Article'].lower().strip().split())
            counter_chinese.update(jieba.lcut(sample['CrossLingualSummary']))

    shared_dictionary_file = open('SharedDictionary.vocab', 'w', encoding='UTF-8')

    items = counter_english.most_common()
    with open('EnglishDictionary.vocab', 'w', encoding='UTF-8') as file:
        for word, _ in items[0:10000]:
            file.write(word + '\n')
            shared_dictionary_file.write(word + '\n')

    items = counter_chinese.most_common()
    with open('ChineseDictionary.vocab', 'w', encoding='UTF-8') as file:
        for word, _ in items:
            file.write(word + '\n')
            shared_dictionary_file.write(word + '\n')
    shared_dictionary_file.close()
