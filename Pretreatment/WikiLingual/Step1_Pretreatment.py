import pickle
import json
import tqdm

if __name__ == '__main__':
    english_data = pickle.load(open('D:/ProjectData/pkl/pkl/english.pkl', 'rb'))
    chinese_data = pickle.load(open('D:/ProjectData/pkl/pkl/chinese.pkl', 'rb'))

    total_sample = []
    for treat_sample in tqdm.tqdm(list(chinese_data.items())):
        chinese_url = treat_sample[0]
        treat_sample = treat_sample[1]
        for section_name in treat_sample:
            english_correlated = english_data[treat_sample[section_name]['english_url']][
                treat_sample[section_name]['english_section_name']]

            current_sample = {'EnglishSummary': english_correlated['summary'],
                              'EnglishDocument': english_correlated['document'],
                              'EnglishSectionName': treat_sample[section_name]['english_section_name'],
                              'EnglishURL': treat_sample[section_name]['english_url'],
                              'ChineseSummary': treat_sample[section_name]['summary'],
                              'ChineseDocument': treat_sample[section_name]['document'],
                              'ChineseSectionName': section_name,
                              'ChineseURL': chinese_url}

            total_sample.append(current_sample)
    json.dump(total_sample, open('WikiLingual_En2ZH.json', 'w', encoding='UTF-8'))
