from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_WikiLingual import build_wiki_lingual
from Tools import ProgressBar
import torch
import os
import datetime

if __name__ == '__main__':
    spanish_train_data, spanish_val_data = build_wiki_lingual(part_name='English2Spanish')
    portuguese_train_data, portuguese_val_data = build_wiki_lingual(part_name='English2Portuguese')
    save_path = 'mt5-small-WikiLingual-Both2English/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/mt5-small')
    optimizer = torch.optim.AdamW(model.parameters(), 1E-4)
    model = model.cuda()

    pbar = ProgressBar(n_total=20 * len(spanish_train_data))
    step_counter = 0
    total_loss = 0.0

    for epoch in range(20):
        for sample_index in range(max(len(spanish_train_data), len(portuguese_train_data))):
            step_counter += 1

            for target_lingual in ['Spanish', 'Portuguese']:
                if sample_index > len(spanish_train_data) and target_lingual == 'Spanish': continue
                if sample_index > len(portuguese_train_data) and target_lingual == 'Portuguese': continue

                if target_lingual == 'Spanish': sample = spanish_train_data[sample_index]
                if target_lingual == 'Portuguese': sample = portuguese_train_data[sample_index]

                article = sample['%sDocument' % target_lingual]
                cross_lingual_summary = sample['EnglishSummary']
                article_tokenized = tokenizer.encode(
                    'summarize %s to English : ' % target_lingual + article, return_tensors='pt', max_length=1024)

                cross_lingual_summary_tokenized = tokenizer.encode(
                    ' ' + cross_lingual_summary, return_tensors='pt', max_length=1024)
                result = model.forward(input_ids=article_tokenized.cuda(),
                                       labels=cross_lingual_summary_tokenized.cuda())
                loss = result.loss

                loss.backward()
                total_loss += loss.data
                optimizer.step()
                model.zero_grad()

            pbar(step_counter, {'loss': loss.data})
            if step_counter % 5000 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))

                with torch.set_grad_enabled(False):
                    val_pbar = ProgressBar(n_total=len(spanish_val_data) + len(portuguese_val_data))

                    val_counter = 0
                    for target_lingual in ['Spanish', 'Portuguese']:
                        total_loss = 0.0
                        if target_lingual == 'Spanish': val_data = spanish_val_data
                        if target_lingual == 'Portuguese': val_data = portuguese_val_data
                        for sample in val_data:
                            val_counter += 1

                            article = sample['%sDocument' % target_lingual]
                            cross_lingual_summary = sample['EnglishSummary']
                            article_tokenized = tokenizer.encode(
                                'summarize %s to English : ' % target_lingual + article, return_tensors='pt',
                                max_length=1024)

                            cross_lingual_summary_tokenized = tokenizer.encode(
                                ' ' + cross_lingual_summary, return_tensors='pt', max_length=1024)
                            result = model.forward(input_ids=article_tokenized.cuda(),
                                                   labels=cross_lingual_summary_tokenized.cuda())
                            loss = result.loss

                            val_pbar(val_counter, {'loss': loss.data})
                            total_loss += loss.item()
                        print('\nSpanish Val Part Loss = ', total_loss)
                        with open(os.path.join(save_path, "log"), "a", encoding="UTF-8") as log:
                            log.write("%s\t step: %6d\t %s loss: %.2f\t \n" % (
                                datetime.datetime.now(), step_counter, target_lingual, total_loss))

                    #################################################

                    filename = "checkpoint-step-%06d" % step_counter
                    full_filename = os.path.join(save_path, filename)
                    model.save_pretrained(save_path + filename)

                total_loss = 0.0
