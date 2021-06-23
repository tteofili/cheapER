import logging
import os

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from cheaper.emt.logging_customized import setup_logging
from cheaper.emt.model import save_model, load_model

setup_logging()


def train(device,
          train_dataloader,
          model,
          optimizer,
          scheduler,
          evaluation,
          num_epocs,
          max_grad_norm,
          save_model_after_epoch,
          experiment_name,
          output_dir,
          model_type,
          silent):
    logging.info("***** Run training *****")
    tb_writer = SummaryWriter(os.path.join(output_dir, experiment_name))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1, silent)
    # for key, value in eval_results.items():
    #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

    f1_top = 0
    best_model_location = None
    best_eval = None
    for epoch in trange(int(num_epocs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=silent)):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if model_type != 'distilbert-base-uncased':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert-base-uncased', 'xlnet-base-cased'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss

        eval_results = evaluation.evaluate(model, device, epoch, silent)
        # for key, value in eval_results.items():
        #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        l1 = eval_results['report'].split('\n')[3].split('       ')[2].split('      ')
        f1 = float(l1[2])
        if f1 > f1_top or f1_top == 0:
            f1_top = f1
            best_model_location = save_model(model, experiment_name + "_best", output_dir, epoch=epoch)
            best_eval = eval_results

        if save_model_after_epoch:
            save_model(model, experiment_name, output_dir, epoch=epoch)

    logging.info("using best model from {}".format(best_model_location))
    best_model = load_model(best_model_location)

    tb_writer.close()
    return best_model, best_eval
