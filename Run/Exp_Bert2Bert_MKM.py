from transformers import EncoderDecoderModel, BertConfig, BertGenerationEncoder, BertGenerationDecoder
from Loader_NCLS import build_mask_dataset, build_overlap_mask_dataset
from Tools import ProgressBar, SaveModel
import torch
import numpy
import os
import datetime

cuda_flag = True

if __name__ == '__main__':
    field, train_data = build_mask_dataset(sample_number=1000, separate_flag=True, word_flag=False)
    _, val_data = build_mask_dataset(use_part='test', separate_flag=True, word_flag=False)

    save_path = 'Bert2Bert_MKM_Character_1K/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    config = BertConfig.from_pretrained('C:/PythonProject/bert-base-uncased')
    config.pad_token_id = field.pad_id
    config.sep_token_id = field.bos_id
    config.bos_token_id = field.bos_id
    config.eos_token_id = field.eos_id
    config.max_length = 2048
    config.max_position_embeddings = 2048
    config.decoder_start_token_id = field.bos_id
    config.vocab_size = len(field.vocab)

    encoder = BertGenerationEncoder(config)
    decoder = BertGenerationDecoder(config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1E-4)

    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))

    for epoch in range(20):
        for i, batch in enumerate(train_data):
            step_counter += 1
            summary = torch.LongTensor(batch.summary).unsqueeze(0)
            article = torch.LongTensor(batch.article).unsqueeze(0)
            labels = torch.LongTensor(batch.label).unsqueeze(0)
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
