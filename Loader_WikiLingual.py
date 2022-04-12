import json
from Loader_CNNDM import CollateClass, DataLoader


def build_wiki_lingual(part_name='En2ZH'):
    load_path = 'D:/PythonProject/Study202203Restart/Pretreatment/WikiLingual_%s.json' % part_name
    treat_data = json.load(open(load_path, 'r', encoding='UTF-8'))
    return treat_data[0:int(len(treat_data) * 0.9)], treat_data[int(len(treat_data) * 0.9):]


# def build_wiki_lingual(part_name='En2ZH', tokenizer=None):
#     load_path = 'D:/PythonProject/Study202203Restart/Pretreatment/WikiLingual_%s.json' % part_name
#
#     treat_data = json.load(open(load_path, 'r', encoding='UTF-8'))
#     collate = CollateClass(tokenizer, cross_lingual_flag=True)
#
#     total_data = []
#     for sample in treat_data:
#         total_data.append({'article': sample['EnglishArticle'], 'summary': sample['Summary'],
#                            'cross_lingual_summary': sample['CrossLingualSummary']})
#         exit()


if __name__ == '__main__':
    build_wiki_lingual()
