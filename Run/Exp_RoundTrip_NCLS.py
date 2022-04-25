import os
import numpy
import torch
import datetime
from Tools import ProgressBar
from Loader_NCLS_Neo import ncls_loader_EN2ZH, build_mask_ncls_treat_sample, build_masked_article_label
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

loss_weight = 5

if __name__ == '__main__':
    train_data = ncls_loader_EN2ZH(sample_number=1000, use_part='train')
    val_data = ncls_loader_EN2ZH(use_part='test')
    save_path = 'mt5-small-NCLS-EN2ZH-RoundTrip/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    summary_model = MT5ForConditionalGeneration.from_pretrained(
        'D:/PythonProject/Study202203Restart/mT5-NCLS-EN2ZH')
    masked_keywords_model = MT5ForConditionalGeneration.from_pretrained(
        'D:\PythonProject\Study202203Restart\mT5_MKM_NCLS')
    masked_keywords_model.eval()
    optimizer = torch.optim.AdamW(summary_model.parameters(), 1E-5)

    summary_model.cuda()
    masked_keywords_model.cuda()

    pbar = ProgressBar(n_total=20 * len(train_data))
    step_counter = 0
    total_loss = 0.0

    for epoch in range(20):
        for sample in train_data:
            step_counter += 1
            article, summary, cross_lingual_summary = sample['Article'], sample['Summary'], sample[
                'CrossLingualSummary']
            article_tokens = tokenizer.encode(
                'summarize English to Chinese : ' + article, return_tensors='pt', max_length=2048)
            cross_lingual_summary_tokens = tokenizer.encode_plus(cross_lingual_summary, return_tensors='pt')[
                'input_ids']

            result = summary_model(input_ids=article_tokens.cuda(), labels=cross_lingual_summary_tokens.cuda())
            loss = result.loss
            logits = result.logits.argmax(dim=-1)

            masked_article, labels = build_masked_article_label(sample)
            ids = tokenizer.encode_plus(masked_article)['input_ids']
            current_input_ids = torch.cat(
                [tokenizer.encode_plus('cross lingual summary : ', add_special_tokens=False, return_tensors='pt')[
                     'input_ids'].cuda(), logits,
                 tokenizer.encode_plus('context : ' + masked_article, return_tensors='pt', max_length=2048)[
                     'input_ids'].cuda()],
                dim=1)
            labels_ids = tokenizer.encode_plus(labels, return_tensors='pt')['input_ids']
            mlm_loss = masked_keywords_model(input_ids=current_input_ids, labels=labels_ids.cuda()).loss * loss_weight

            pbar(step_counter, {'loss': loss.item(), 'mkm_loss': mlm_loss.item()})
            loss += mlm_loss

            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            summary_model.zero_grad()

            if step_counter % 1 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                summary_model.eval()
                with torch.set_grad_enabled(False):
                    val_pbar = ProgressBar(n_total=len(val_data))
                    for i, sample in enumerate(val_data):
                        article, summary, cross_lingual_summary = sample['Article'], sample['Summary'], sample[
                            'CrossLingualSummary']
                        article_tokens = tokenizer.encode(
                            'summarize English to Chinese : ' + article, return_tensors='pt', max_length=2048)
                        cross_lingual_summary_tokens = \
                            tokenizer.encode_plus(cross_lingual_summary, return_tensors='pt')['input_ids']

                        result = summary_model(input_ids=article_tokens, labels=cross_lingual_summary_tokens)
                        loss = result.loss

                        val_pbar(i, {'loss': loss.data})
                        total_loss += loss.item()
                    print('\nVal Part Loss = ', total_loss)
                    with open(os.path.join(save_path, "log"), "a", encoding="UTF-8") as log:
                        log.write(
                            "%s\t step: %6d\t loss: %.2f\t \n" % (datetime.datetime.now(), step_counter, total_loss))

                    filename = "checkpoint-step-%06d" % step_counter
                    full_filename = os.path.join(save_path, filename)
                    summary_model.save_pretrained(save_path + filename)
                summary_model.train()
                total_loss = 0.0
