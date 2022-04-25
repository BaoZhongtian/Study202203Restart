import jieba
import tqdm
import collections
from Loader_NCLS import ncls_loader_EN2ZH

if __name__ == '__main__':
    # counter_chinese = collections.Counter()
    #
    # for use_part in ['valid', 'test', 'train']:
    #     result = ncls_loader(use_part=use_part)
    #     print()
    #     for sample in tqdm.tqdm(result):
    #         counter_chinese.update([_ for _ in sample['CrossLingualSummary'].lower()])
    #
    # items = counter_chinese.most_common()
    # with open('ChineseDictionary_Character.vocab', 'w', encoding='UTF-8') as file:
    #     for word, _ in items:
    #         file.write(word + '\n')

    total_dictionary = []
    with open('../EnglishDictionary.vocab', 'r', encoding='UTF-8') as file:
        data = file.readlines()
        total_dictionary.extend([_.replace('\n', '') for _ in data])

    with open('../ChineseDictionary_Character.vocab', 'r', encoding='UTF-8') as file:
        data = file.readlines()
        total_dictionary.extend([_.replace('\n', '') for _ in data])
    with open('../SharedDictionary_Character.vocab', 'w', encoding='UTF-8') as file:
        for sample in total_dictionary:
            file.write(sample + '\n')
