from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from Loader_CNNDM import loader_cnndm
from Loader_NCLS_Neo import build_ncls_neo, build_mask_ncls_for_mt5, build_mask_ncls_treat_sample
from Tools import ProgressBar, SaveModel
import torch
import numpy
import os
import datetime

cuda_flag = True

if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/mt5-small')
    train_data, val_data = build_mask_ncls_for_mt5(tokenizer=tokenizer, sample_number=1000)

    save_path = 'mT5_MKM_NCLS/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))

    for epoch in range(20):
        for i, batch in enumerate(train_data):
            if batch is None: continue
            step_counter += 1

            batch = build_mask_ncls_treat_sample(batch, tokenizer)
            if batch is None: continue
            input_ids, labels = batch['input_ids'], batch['labels']
            print(tokenizer.batch_decode(input_ids))
            print(tokenizer.batch_decode(labels))
            exit()
            if cuda_flag:
                input_ids, labels = input_ids.cuda(), labels.cuda()
            loss = model.forward(input_ids=input_ids, labels=labels).loss
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
                        batch = build_mask_ncls_treat_sample(batch, tokenizer)
                        if batch is None: continue

                        input_ids, labels = batch['input_ids'], batch['labels']
                        if cuda_flag:
                            input_ids, labels = input_ids.cuda(), labels.cuda()
                        loss = model.forward(input_ids=input_ids, labels=labels).loss

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
