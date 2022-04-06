from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_NCLS import ncls_loader

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/Study202203Restart/checkpoint-step-070000/')

    val_data = ncls_loader(use_part='test')

    for sample in val_data:
        article = sample['Article']
        summary = sample['Summary']
        cross_lingual_summary = sample['CrossLingualSummary']
        article_tokenized = tokenizer.encode(
            'summarize English to Chinese : ' + article, return_tensors='pt', max_length=512)
        summary_tokenized = tokenizer.encode(' ' + summary, return_tensors='pt', max_length=512)
        cross_lingual_summary_tokenized = tokenizer.encode(
            ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)

        result = model.generate(article_tokenized, min_length=48, max_length=128, num_beams=16)

        print(tokenizer.batch_decode(result))
        print(cross_lingual_summary)
        exit()
