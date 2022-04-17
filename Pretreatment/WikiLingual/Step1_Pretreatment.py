import pickle
import json
import tqdm

if __name__ == '__main__':
    target_lingual = 'Spanish'
    english_data = pickle.load(open('D:/ProjectData/pkl/pkl/english.pkl', 'rb'))
    target_data = pickle.load(open('D:/ProjectData/pkl/pkl/%s.pkl' % target_lingual, 'rb'))

    total_sample = []
    for treat_sample in tqdm.tqdm(list(target_data.items())):
        chinese_url = treat_sample[0]
        treat_sample = treat_sample[1]
        for section_name in treat_sample:
            english_correlated = english_data[treat_sample[section_name]['english_url']][
                treat_sample[section_name]['english_section_name']]

            current_sample = {'EnglishSummary': english_correlated['summary'],
                              'EnglishDocument': english_correlated['document'],
                              'EnglishSectionName': treat_sample[section_name]['english_section_name'],
                              'EnglishURL': treat_sample[section_name]['english_url'],
                              '%sSummary' % target_lingual: treat_sample[section_name]['summary'],
                              '%sDocument' % target_lingual: treat_sample[section_name]['document'],
                              '%sSectionName' % target_lingual: section_name,
                              '%sURL' % target_lingual: chinese_url}
            total_sample.append(current_sample)
    json.dump(total_sample, open('WikiLingual_English2%s.json' % target_lingual, 'w', encoding='UTF-8'))
