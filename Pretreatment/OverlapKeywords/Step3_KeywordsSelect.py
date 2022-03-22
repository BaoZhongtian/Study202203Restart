import tqdm
from Loader_NCLS import ncls_loader

if __name__ == '__main__':
    use_part = 'valid'
    ignore_number = 50
    select_words = 10
    sample_number = 100

    ignore_words = set()
    with open('IgnoreWords.txt', 'r', encoding='UTF-8') as file:
        for _ in range(ignore_number):
            raw_text = file.readline()
            ignore_words.add(raw_text.split(',')[0])

    train_data = ncls_loader(use_part=use_part, sample_number=sample_number)
    for sample in tqdm.tqdm(train_data):
        article = sample['Article'].lower().strip().split()
        summary = set(sample['Summary'].lower().strip().split())
        for word in ignore_words:
            if word in summary: summary.remove(word)

        keywords_dictionary = {}
        for word in article:
            if word in summary:
                if word in keywords_dictionary.keys():
                    keywords_dictionary[word] += 1
                else:
                    keywords_dictionary[word] = 1
        keywords_tuple = [[_, keywords_dictionary[_]] for _ in keywords_dictionary]
        keywords_tuple = sorted(keywords_tuple, key=lambda x: x[-1], reverse=True)[0:select_words]
        keywords_select = set([_[0] for _ in keywords_tuple])
        print(keywords_select)
        exit()
