import os
import json
import tqdm

if __name__ == '__main__':
    load_path = 'D:/ProjectData/NCLS_MKM_Result'
    accurate_counter, total_counter = 0, 0
    for filename in tqdm.tqdm(os.listdir(load_path)):
        treat_data = json.load(open(os.path.join(load_path, filename), 'r'))

        for index in range(len(treat_data['logits'])):
            total_counter += 1
            if treat_data['logits'][index] == treat_data['labels'][index]: accurate_counter += 1
    print(accurate_counter, total_counter)
    print(accurate_counter * 1.0 / total_counter)
