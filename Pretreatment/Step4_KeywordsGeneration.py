import os
import numpy
import joblib
import tqdm
import json

if __name__ == '__main__':
    with open('EnglishDictionary.vocab', 'r', encoding='UTF-8') as file:
        data = file.readlines()
    id2word = {}
    for index in range(len(data)):
        id2word[index] = data[index].replace('\n', '')

    tfidf = joblib.load('TfIdfTransformer.joblib')
    load_path = 'E:/ProjectData/NCLS/Matrix/'
    save_path = 'E:/ProjectData/NCLS/Keywords-Result/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    for filename in os.listdir(load_path):
        if os.path.exists(save_path + filename.replace('csv', 'json')): continue
        with open(save_path + filename.replace('csv', 'json'), 'w'):
            pass
        print(filename)

        with open(load_path + filename, 'r') as file:
            raw_data = file.readlines()
        batch_data = []
        for sample in raw_data:
            batch_data.append([int(_) for _ in sample.replace('\n', '').split(',')])

        treated_data = tfidf.transform(batch_data)

        total_keywords = []
        for index in tqdm.trange(len(batch_data)):
            sample_keywords = []
            for _ in range(50):
                sample_keywords.append([id2word[numpy.argmax(treated_data[index])],
                                        treated_data[index, numpy.argmax(treated_data[index])]])
                treated_data[index, numpy.argmax(treated_data[index])] = 0.0
            total_keywords.append(sample_keywords)
        json.dump(total_keywords, open(save_path + filename.replace('csv', 'json'), 'w'))

        # exit()
        # with open(save_path+filename,'w') as file:
