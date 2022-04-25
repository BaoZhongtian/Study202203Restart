import json
import tqdm
import collections
from Loader_NCLS import ncls_loader_EN2ZH

if __name__ == '__main__':
    counter_english = collections.Counter()
    for use_part in ['test', 'train']:
        raw_data = json.load(open('D:/PythonProject/Study202203Restart/Pretreatment/CNNDM_%s.json' % use_part))

        for sample in tqdm.tqdm(raw_data):
            counter_english.update(sample['article'].lower().strip().split())

    items = counter_english.most_common()
    with open('../CNNDM_Dictionary.vocab', 'w', encoding='UTF-8') as file:
        for word, _ in items[0:10000]:
            file.write(word + '\n')
