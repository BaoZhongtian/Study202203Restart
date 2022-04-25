from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_NCLS import ncls_loader_EN2ZH
from Loader_WikiLingual import build_wiki_lingual
import os

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained(
        'D:/PythonProject/Study202203Restart/mt5-small-WikiLingual-English2Portuguese/checkpoint-step-230000/')
    # val_data = ncls_loader(use_part='test')
    train_data, val_data = build_wiki_lingual(part_name='English2Portuguese')

    save_path = 'WikiLingualResult-English2Portuguese/'
    for index, sample in enumerate(val_data):
        print('\rTreating %d' % index, end='')
        if os.path.exists(save_path + '%08d.csv' % index): continue
        with open(save_path + '%08d.csv' % index, 'w') as file:
            pass

        article = sample['PortugueseDocument']
        cross_lingual_summary = sample['EnglishSummary']
        article_tokenized = tokenizer.encode(
            'summarize Portuguese to English : ' + article, return_tensors='pt', max_length=512)
        cross_lingual_summary_tokenized = tokenizer.encode(
            ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)

        result = model.generate(article_tokenized, min_length=int(0.75 * len(cross_lingual_summary_tokenized[0])),
                                max_length=int(1.5 * len(cross_lingual_summary_tokenized[0])),
                                num_beams=16, repetition_penalty=5.0).detach().cpu().numpy()
        print(tokenizer.batch_decode(result, skip_special_tokens=True))
        print(cross_lingual_summary)
        exit()

        # with open(save_path + '%08d.csv' % index, 'w') as file:
        #     file.write(tokenizer.batch_decode(result, skip_special_tokens=True)[0])
