from transformers import EncoderDecoderModel, BertConfig, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, \
    BertModel
from Loader_CNNDM import loader_cnndm
from Loader_NCLS_Neo import build_ncls_neo
from Tools import ProgressBar, SaveModel
import torch
import numpy
import os
import datetime

cuda_flag = True

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(
        'D:/PythonProject/bert-base-multilingual-cased/')
    # train_data, val_data = loader_cnndm(
    #     batch_size=4, tokenizer=tokenizer, small_data_flag=False, train_part_shuffle=True)
    train_data, val_data = build_ncls_neo(
        tokenizer=tokenizer, train_part_shuffle=True, sample_number=1000)

    save_path = 'Bert2Bert_MKM_NCLS/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'D:/PythonProject/bert-base-multilingual-cased/',
        'D:/PythonProject/bert-base-multilingual-cased/')
    # model = EncoderDecoderModel.from_pretrained(
    #     '/root/autodl-tmp/MaskedKeywords/MaskedKeywordsExperiment/Bert2Bert_MKM_CNNDM_BertBaseUncased/checkpoint-step-015000')
    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    total_loss = 0.0
    step_counter = 15000
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))

    for epoch in range(20):
        for i, batch in enumerate(train_data):
            if batch is None: continue
            step_counter += 1

            summary = torch.LongTensor(batch.summary)
            article = torch.LongTensor(batch.article)
            labels = torch.LongTensor(batch.label)
            if cuda_flag:
                summary, article, labels = summary.cuda(), article.cuda(), labels.cuda()

            loss = model.forward(input_ids=summary, decoder_input_ids=article, labels=labels).loss
            loss.backward()
            total_loss += loss.data

            optimizer.step()
            model.zero_grad()
            pbar(epoch * len(train_data) + i, {'loss': loss.data})
            if step_counter % 5000 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                with torch.set_grad_enabled(False):
                    val_pbar = ProgressBar(n_total=len(val_data))
                    for i, batch in enumerate(val_data):
                        if batch is None: continue
                        if len(batch.article) == 0: continue

                        summary = torch.LongTensor(batch.summary)
                        article = torch.LongTensor(batch.article)
                        labels = torch.LongTensor(batch.label)
                        if cuda_flag:
                            summary, article, labels = summary.cuda(), article.cuda(), labels.cuda()
                        loss = model.forward(input_ids=summary, decoder_input_ids=article, labels=labels).loss

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
