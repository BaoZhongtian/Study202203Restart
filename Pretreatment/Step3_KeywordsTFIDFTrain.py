import os
import numpy
import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

if __name__ == '__main__':
    load_path = 'E:/ProjectData/NCLS/Matrix/'
    transformer = TfidfTransformer()
    total_data = []
    for filename in tqdm.tqdm(os.listdir(load_path)[0:10]):
        with open(load_path + filename, 'r') as file:
            raw_data = file.readlines()

        for sample in raw_data:
            total_data.append([int(_) for _ in sample.replace('\n', '').split(',')])

        print('\n', numpy.shape(total_data))

    print(numpy.shape(total_data))
    tfidf = transformer.fit_transform(total_data)
    joblib.dump(transformer, 'TfIdfTransformer.joblib')
