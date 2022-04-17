from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_NCLS import ncls_loader
from Loader_WikiLingual import build_wiki_lingual
import os

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    input_ids = tokenizer.encode('The <extra_id_1> walks in <extra_id_2> park', return_tensors='pt')
    lm_labels = tokenizer.encode('<extra_id_1> cute dog <extra_id_2> the <extra_id_3> </s>', return_tensors='pt')
    print(input_ids)
    print(lm_labels)
    exit()

    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/Study202203Restart/checkpoint-step-075000/')
    # val_data = ncls_loader(use_part='test')
    train_data, val_data = build_wiki_lingual()

    save_path = 'WikiLingualResult-075000/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    for index, sample in enumerate(val_data):
        print('\rTreating %d' % index, end='')
        # article = sample['Article']
        # summary = sample['Summary']
        # cross_lingual_summary = sample['CrossLingualSummary']
        # article_tokenized = tokenizer.encode(
        #     'summarize English to Chinese : ' + article, return_tensors='pt', max_length=512)
        # summary_tokenized = tokenizer.encode(' ' + summary, return_tensors='pt', max_length=512)
        # cross_lingual_summary_tokenized = tokenizer.encode(
        #     ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)

        article = sample['EnglishDocument']
        cross_lingual_summary = sample['ChineseSummary']
        article_tokenized = tokenizer.encode(
            'summarize English to Chinese : ' + article, return_tensors='pt', max_length=512)
        cross_lingual_summary_tokenized = tokenizer.encode(
            ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)

        result = model.generate(article_tokenized, min_length=len(cross_lingual_summary_tokenized),
                                max_length=int(1.5 * len(cross_lingual_summary_tokenized)),
                                num_beams=16).detach().cpu().numpy()

        with open(save_path + '%08d.csv' % index, 'w') as file:
            file.write(tokenizer.batch_decode(result, skip_special_tokens=True)[0])
