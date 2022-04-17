import numpy
import tqdm
from Loader_NCLS import ncls_loader
from Loader_CNNDM import CollateClass, DataLoader


def build_ncls_neo(sample_number=None, tokenizer=None, train_part_shuffle=False):
    train_data_raw = ncls_loader(sample_number, use_part='train')
    test_data_raw = ncls_loader(use_part='test')

    collate = CollateClass(tokenizer, cross_lingual_flag=True)

    train_data, test_data = [], []
    for sample in train_data_raw:
        train_data.append({'article': sample['Article'], 'summary': sample['Summary'],
                           'cross_lingual_summary': sample['CrossLingualSummary']})
    for sample in test_data_raw:
        test_data.append({'article': sample['Article'], 'summary': sample['Summary'],
                          'cross_lingual_summary': sample['CrossLingualSummary']})

    train_dataset = DataLoader(train_data, batch_size=1, shuffle=train_part_shuffle, collate_fn=collate.collate)
    test_dataset = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate.collate)

    return train_dataset, test_dataset


ignore_words = set()
with open('D:/PythonProject/Study202203Restart/Pretreatment/IgnoreWords.txt', 'r', encoding='UTF-8') as file:
    for _ in range(50):
        raw_text = file.readline()
        ignore_words.add(raw_text.split(',')[0])


def build_mask_ncls_treat_sample(sample, tokenizer, keywords_number=10):
    keywords_dictionary = {}
    for word in sample['Article'].lower().strip().split():
        if word in sample['Summary'].lower().strip().split():
            if word in ignore_words: continue
            if word in keywords_dictionary.keys():
                keywords_dictionary[word] += 1
            else:
                keywords_dictionary[word] = 1

    keywords_tuple = [[_, keywords_dictionary[_]] for _ in keywords_dictionary]
    keywords_tuple = sorted(keywords_tuple, key=lambda x: x[-1], reverse=True)[0:keywords_number]
    treat_keywords = set([_[0] for _ in keywords_tuple])
    if len(treat_keywords) == 0: return None
    treat_keywords = [_ for _ in treat_keywords]

    current_lm_sentence = ''
    for index in range(len(treat_keywords)):
        current_lm_sentence += '<extra_id_%d> ' % (index + 1) + treat_keywords[index] + ' '
    current_lm_sentence += '<extra_id_%d>' % (len(treat_keywords) + 1)

    current_article = sample['Article'].lower()
    for index in range(len(treat_keywords)):
        current_article = current_article.replace(treat_keywords[index], '<extra_id_%d>' % (index + 1))
    current_input_ids = 'cross lingual summary : ' + sample[
        'CrossLingualSummary'] + ' context : ' + current_article

    return {'input_ids': tokenizer.encode_plus(current_input_ids, return_tensors='pt', max_length=2048)['input_ids'],
            'labels': tokenizer.encode_plus(current_lm_sentence, return_tensors='pt')['input_ids']}


def build_mask_ncls_for_mt5(sample_number=None, tokenizer=None, ignore_number=50):
    train_data = ncls_loader(sample_number, use_part='train')
    test_data = ncls_loader(use_part='test')
    return train_data, test_data


if __name__ == '__main__':
    import transformers

    tokenizer = transformers.MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    build_mask_ncls_for_mt5(sample_number=100, tokenizer=tokenizer)
