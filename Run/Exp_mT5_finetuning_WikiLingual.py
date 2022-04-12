from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_WikiLingual import build_wiki_lingual
from Tools import ProgressBar
import torch
import os
import datetime

if __name__ == '__main__':
    train_data, val_data = build_wiki_lingual()
    save_path = 'mt5-small-WikiLingual-EN2ZH/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    tokenizer = MT5Tokenizer.from_pretrained('/root/autodl-tmp/MaskedKeywords/MaskedKeywordsExperiment/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained(
        '/root/autodl-tmp/MaskedKeywords/MaskedKeywordsExperiment/mt5-small')
    optimizer = torch.optim.AdamW(model.parameters(), 1E-4)
    model = model.cuda()

    pbar = ProgressBar(n_total=20 * len(train_data))
    step_counter = 0
    total_loss = 0.0

    for epoch in range(20):
        for sample in train_data:
            step_counter += 1

            article = sample['EnglishDocument']
            cross_lingual_summary = sample['ChineseSummary']
            article_tokenized = tokenizer.encode(
                'summarize English to Chinese : ' + article, return_tensors='pt', max_length=512)
            # article = sample['ChineseDocument']
            # cross_lingual_summary = sample['EnglishSummary']
            # article_tokenized = tokenizer.encode(
            #    'summarize Chinese to English : ' + article, return_tensors='pt', max_length=512)

            cross_lingual_summary_tokenized = tokenizer.encode(
                ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)
            result = model.forward(input_ids=article_tokenized.cuda(), labels=cross_lingual_summary_tokenized.cuda())
            loss = result.loss

            loss.backward()
            total_loss += loss.data
            optimizer.step()
            model.zero_grad()

            pbar(step_counter, {'loss': loss.data})
            if step_counter % 5000 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                with torch.set_grad_enabled(False):
                    val_pbar = ProgressBar(n_total=len(val_data))
                    for i, batch in enumerate(val_data):
                        article = batch['EnglishDocument']
                        cross_lingual_summary = batch['ChineseSummary']
                        article_tokenized = tokenizer.encode(
                            'summarize English to Chinese : ' + article, return_tensors='pt', max_length=512)
                        # article = batch['ChineseDocument']
                        # cross_lingual_summary = batch['EnglishSummary']
                        # article_tokenized = tokenizer.encode(
                        #    'summarize Chinese to English : ' + article, return_tensors='pt', max_length=512)

                        cross_lingual_summary_tokenized = tokenizer.encode(
                            ' ' + cross_lingual_summary, return_tensors='pt', max_length=512)
                        result = model.forward(input_ids=article_tokenized.cuda(),
                                               labels=cross_lingual_summary_tokenized.cuda())
                        loss = result.loss

                        val_pbar(i, {'loss': loss.data})
                        total_loss += loss.item()
                    print('\nVal Part Loss = ', total_loss)
                    with open(os.path.join(save_path, "log"), "a", encoding="UTF-8") as log:
                        log.write(
                            "%s\t step: %6d\t loss: %.2f\t \n" % (datetime.datetime.now(), step_counter, total_loss))

                    filename = "checkpoint-step-%06d" % step_counter
                    full_filename = os.path.join(save_path, filename)
                    model.save_pretrained(save_path + filename)

                total_loss = 0.0
