from transformers import EncoderDecoderModel, BertConfig, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer
from Loader_CNNDM import loader_cnndm
from Tools import ProgressBar, SaveModel
import torch
import numpy
import os
import datetime

cuda_flag = True

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('D:/PythonProject/bert-base-uncased/')
    train_data, val_data = loader_cnndm(
        batch_size=3, tokenizer=tokenizer, small_data_flag=True, train_part_shuffle=False)

    save_path = 'Bert2Bert_MKM_CNNDM_BertBaseUncased/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    encoder = BertGenerationEncoder.from_pretrained('D:/PythonProject/bert-base-uncased/')
    decoder = BertGenerationDecoder.from_pretrained('D:/PythonProject/bert-base-uncased/')
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))

    for epoch in range(20):
        for i, batch in enumerate(train_data):
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
                        summary = torch.LongTensor(batch.summary).unsqueeze(0)
                        article = torch.LongTensor(batch.article).unsqueeze(0)
                        labels = torch.LongTensor(batch.label).unsqueeze(0)
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
                    SaveModel(model, save_path + filename)
                total_loss = 0.0
