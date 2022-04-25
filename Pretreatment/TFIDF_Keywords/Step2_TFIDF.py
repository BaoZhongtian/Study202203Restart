import os
import numpy
import tqdm
from Loader_NCLS import ncls_loader_EN2ZH

if __name__ == '__main__':
    with open('../EnglishDictionary.vocab', 'r', encoding='UTF-8') as file:
        data = file.readlines()

    english_dictionary = set([_.replace('\n', '') for _ in data])
    word2id = {}
    for index in range(len(data)):
        word2id[data[index].replace('\n', '')] = index

    train_data = [_['Article'].lower().strip().split()[0:2048] for _ in
                  ncls_loader_EN2ZH(use_part='valid')]

    matrix = []
    match_counter, total_counter = 0, 0

    save_path = 'E:/ProjectData/NCLS/Matrix-Valid/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    for start_position in tqdm.trange(0, len(train_data), 5000):
        if os.path.exists(save_path + '%08d.csv' % start_position): continue
        with open(save_path + '%08d.csv' % start_position, 'w'):
            pass

        batch_data = train_data[start_position:start_position + 5000]
        with open(save_path + '%08d.csv' % start_position, 'w') as file:
            for indexX in range(len(batch_data)):
                line = numpy.zeros(len(english_dictionary))
                for indexY in range(len(batch_data[indexX])):
                    if batch_data[indexX][indexY] in english_dictionary:
                        line[word2id[batch_data[indexX][indexY]]] += 1
                for indexY in range(len(line)):
                    if indexY != 0: file.write(',')
                    file.write(str(int(line[indexY])))
                file.write('\n')
