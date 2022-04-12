import numpy
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


if __name__ == '__main__':
    import transformers

    tokenizer = transformers.BertTokenizer.from_pretrained('D:/PythonProject/bert-base-multilingual-cased')
    build_ncls_neo(sample_number=100, tokenizer=tokenizer)
