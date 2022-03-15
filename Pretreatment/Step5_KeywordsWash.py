import os
import json
import tqdm

ignore_most_frequent = 50
select_keywords = 20

if __name__ == '__main__':
    # load_path = 'E:/ProjectData/NCLS/Keywords-Result/'
    #
    # repeat_dictionary = {}
    # for filename in tqdm.tqdm(os.listdir(load_path)):
    #     treat_data = json.load(open(os.path.join(load_path, filename), 'r'))
    #     for indexX in range(len(treat_data)):
    #         for indexY in range(20):
    #             if treat_data[indexX][indexY][0] in repeat_dictionary.keys():
    #                 repeat_dictionary[treat_data[indexX][indexY][0]] += 1
    #             else:
    #                 repeat_dictionary[treat_data[indexX][indexY][0]] = 1
    #
    # result = [[key, repeat_dictionary[key]] for key in repeat_dictionary.keys()]
    # result = sorted(result, key=lambda x: x[-1], reverse=True)
    # with open('RepeatWords.txt', 'w', encoding='UTF-8') as file:
    #     for index in range(ignore_most_frequent):
    #         file.write(result[index][0] + '\n')
    # ignore_words = [_[0] for _ in result[0:ignore_most_frequent]]
    # print(ignore_words)

    # total_keywords = []
    # for filename in tqdm.tqdm(os.listdir(load_path)):
    #     treat_data = json.load(open(os.path.join(load_path, filename), 'r'))
    #     for treat_sample in treat_data:
    #         treat_keywords = []
    #
    #         search_index = -1
    #         while len(treat_keywords) < select_keywords:
    #             search_index += 1
    #             if search_index >= len(treat_sample): break
    #             if treat_sample[search_index][0] in ignore_words: continue
    #             treat_keywords.append(treat_sample[search_index])
    #         total_keywords.append(treat_keywords)
    # json.dump(total_keywords, open('TrainKeywords.json', 'w'))

    with open('RepeatWords.txt', 'r', encoding='UTF-8') as file:
        data = file.readlines()
    ignore_words = [_[0:-1] for _ in data[0:ignore_most_frequent]]

    load_path = 'E:/ProjectData/NCLS/Keywords-Valid-Result/'
    total_keywords = []
    for filename in tqdm.tqdm(os.listdir(load_path)):
        treat_data = json.load(open(os.path.join(load_path, filename), 'r'))
        for treat_sample in treat_data:
            treat_keywords = []

            search_index = -1
            while len(treat_keywords) < select_keywords:
                search_index += 1
                if search_index >= len(treat_sample): break
                if treat_sample[search_index][0] in ignore_words: continue
                treat_keywords.append(treat_sample[search_index])
            total_keywords.append(treat_keywords)
    json.dump(total_keywords, open('ValidKeywords.json', 'w'))
