import logging
import os
import random
import string

import numpy as np
import pandas as pd
from torch import nn

from cheaper.data import deepmatcher_format
from cheaper.emt import prediction
from cheaper.emt.data_loader import load_data, DataType
from cheaper.emt.data_representation import DeepMatcherProcessor
from cheaper.emt.evaluation import Evaluation
from cheaper.emt.logging_customized import setup_logging
from cheaper.emt.model import save_model, load_model
from cheaper.emt.optimizer import build_optimizer
from cheaper.emt.torch_initializer import initialize_gpu_seed
from cheaper.emt.training import train
from transformers import pipeline, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

setup_logging()

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import LineByLineTextDataset
from datasets import load_dataset, load_metric

BATCH_SIZE = 8
MAX_SEQ_LENGTH = 250


class EMTERModel:

    def __init__(self, model_type, model_noise: bool = False, add_layers: int = 0):
        device, n_gpu = initialize_gpu_seed(22)
        self.model_type = model_type
        self.model_noise = model_noise

        self.tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)

        if add_layers > 0:
            config = AutoConfig.from_pretrained(self.model_type)
            no_layers = config.num_hidden_layers + add_layers
            config.num_hidden_layers = no_layers
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_type, config=config).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_type).to(device)

        if model_noise:
            self.model.config.dropout = 0.5
            self.model.config.qa_dropout = 0.5
            self.model.config.attention_dropout = 0.5
            self.model.config.seq_classif_dropout = 0.5

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_type).to(device)
        if device.type == 'cpu':
            device = -1
        else:
            device = 0
        self.noise_pipeline = pipeline('fill-mask', model=self.mlm_model, tokenizer=self.tokenizer, device=device)

    def adaptive_ft(self, unlabelled_train_file, unlabelled_valid_file, dataset_name, model_type,
                    seq_length=MAX_SEQ_LENGTH, epochs=3, lr=5e-5, ow=False):
        device, n_gpu = initialize_gpu_seed(22)
        model_dir = 'models/' + dataset_name + "/mlm-" + model_type
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if os.path.exists(model_dir + '/pytorch_model.bin') and not ow:
            self.mlm_model = load_model(model_dir)
        else:
            self.mlm_model = self.mlm_model.to(device)
            train_dataset = LineByLineTextDataset(
                tokenizer=self.tokenizer,
                file_path=unlabelled_train_file,
                block_size=seq_length,
            )

            valid_dataset = LineByLineTextDataset(
                tokenizer=self.tokenizer,
                file_path=unlabelled_valid_file,
                block_size=seq_length,
            )

            training_args = TrainingArguments(
                learning_rate=lr,
                output_dir='./models/' + dataset_name,  # output directory
                per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
                per_device_eval_batch_size=BATCH_SIZE * 4,  # batch size for evaluation
                logging_dir='./logs',  # directory for storing logs
                save_total_limit=2,
                do_eval=True,
                num_train_epochs=epochs
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
            )

            trainer = Trainer(
                model=self.mlm_model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                data_collator=data_collator,
                train_dataset=train_dataset,  # training dataset
                eval_dataset=valid_dataset  # evaluation dataset
            )

            trainer.train()
            trainer.save_model(model_dir)

    def train(self, label_train, label_valid, model_type, dataset_name, seq_length=MAX_SEQ_LENGTH, warmup=False,
              epochs=3, lr=1e-5, adaptive_ft=False, silent=False, batch_size=BATCH_SIZE, weight_decay=0,
              label_smoothing=0, hf_training=False, best_model='eval_loss'):
        device, n_gpu = initialize_gpu_seed(22)

        if adaptive_ft:
            adaptive_ft_model_dir = 'models/' + dataset_name + "/mlm-" + model_type
            logging.info('loading adaptive_ft model from {}'.format(adaptive_ft_model_dir))
            config = self.model.config
            self.model = load_model(adaptive_ft_model_dir, config=config)

        self.model = self.model.to(device)

        trainF, validF = deepmatcher_format.tofiles(label_train, label_valid, dataset_name)

        if hf_training:
            metric = load_metric("f1")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)

            train_dataset = load_dataset('csv', data_files=trainF, split='train')
            valid_dataset = load_dataset('csv', data_files=validF, split='train')

            def tokenize_function(example):
                text_a = ' '.join({k: str(v) for k, v in example.items() if k.startswith('left_')}.values())
                text_b = ' '.join({k: str(v) for k, v in example.items() if k.startswith('right_')}.values())
                return self.tokenizer(
                    text_a, text_b, padding="max_length", truncation=True, max_length=seq_length
                )

            train_dataset = train_dataset.map(tokenize_function)
            valid_dataset = valid_dataset.map(tokenize_function)

            train_dataset = self.prepare_columns(train_dataset)
            valid_dataset = self.prepare_columns(valid_dataset)

            if warmup:
                warmup_ratio = 0.06
            else:
                warmup_ratio = 0

            metric_for_best_model = best_model
            if metric_for_best_model == 'eval_f1':
                greater_is_better = True
            else:
                greater_is_better = False

            training_args = TrainingArguments(
                learning_rate=lr,
                output_dir='./models/' + dataset_name,  # output directory
                per_device_train_batch_size=batch_size,  # batch size per device during training
                per_device_eval_batch_size=batch_size,  # batch size for evaluation
                logging_dir='./logs',  # directory for storing logs
                do_eval=True,
                num_train_epochs=epochs,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                warmup_ratio=warmup_ratio,
                adam_epsilon=1e-6,
                adam_beta1=0.9,
                adam_beta2=0.98,
                weight_decay=weight_decay,
                logging_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=2,
                greater_is_better=greater_is_better,
                metric_for_best_model=metric_for_best_model,
                max_grad_norm=1.0,
                label_smoothing_factor=label_smoothing
            )

            trainer = Trainer(
                tokenizer=self.tokenizer,
                model=self.model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=train_dataset,  # training dataset
                eval_dataset=valid_dataset,  # evaluation dataset
                compute_metrics=compute_metrics,
            )

            if greater_is_better:
                trainer.add_callback(EarlyStoppingCallback(10))

            trainer.train()
            model_dir = 'models/' + dataset_name
            trainer.save_model(model_dir)
            eval_out = trainer.evaluate(valid_dataset)
            f1 = eval_out['eval_f1']
            return 'nan', 'nan', f1, 'nan', 'nan', 'nan'

        else:
            processor = DeepMatcherProcessor()
            train_examples = processor.get_train_examples_file(trainF)
            label_list = processor.get_labels()
            training_data_loader = load_data(train_examples, label_list, self.tokenizer, seq_length, batch_size,
                                             DataType.TRAINING, self.model_type)

            num_epochs = epochs
            num_train_steps = len(training_data_loader) * num_epochs

            learning_rate = lr
            adam_eps = 1e-6
            if warmup:
                warmup_steps = int(len(training_data_loader) * 0.1)
                weight_decay = 0.01
            else:
                warmup_steps = 0
                weight_decay = 0
            optimizer, scheduler = build_optimizer(self.model, num_train_steps, learning_rate, adam_eps, warmup_steps,
                                                   weight_decay)

            eval_examples = processor.get_test_examples_file(validF)
            evaluation_data_loader = load_data(eval_examples, label_list, self.tokenizer, seq_length, batch_size,
                                               DataType.EVALUATION, self.model_type)

            exp_name = 'models/' + dataset_name
            evaluation = Evaluation(evaluation_data_loader, exp_name, exp_name, len(label_list), self.model_type)

            self.model, result = train(device, training_data_loader, self.model, optimizer, scheduler, evaluation,
                                       num_epochs, 1.0, False, exp_name, exp_name, self.model_type, silent)

            save_model(self.model, exp_name, exp_name, tokenizer=self.tokenizer)

            try:
                l0 = result.split('\n')[2].split('       ')[2].split('      ')
                l1 = result.split('\n')[3].split('       ')[2].split('      ')
            except:
                l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
                l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')

            p = l1[0]
            r = l1[1]
            f1 = l1[2]
            pnm = l0[0]
            rnm = l0[1]
            f1nm = l0[2]
            return p, r, f1, pnm, rnm, f1nm

    def prepare_columns(self, dataset):
        for c in dataset.column_names:
            if c.startswith('left_') or c.startswith('right_') or c.startswith('id'):
                dataset = dataset.remove_columns([c])
            if c == 'label':
                dataset = dataset.rename_column(c, 'labels')
        dataset = dataset.with_format('torch')
        return dataset

    def load(self, path):
        self.model = load_model(path, do_lower_case=True)
        return self

    def save(self, path):
        save_model(self.model, path, path, tokenizer=self.tokenizer)

    def eval(self, label_test, dataset_name, seq_length=MAX_SEQ_LENGTH, silent=False, batch_size=BATCH_SIZE):
        device, n_gpu = initialize_gpu_seed(22)

        self.model = self.model.to(device)

        processor = DeepMatcherProcessor()
        trainF, testF = deepmatcher_format.tofiles(label_test, label_test, dataset_name)
        label_list = processor.get_labels()

        eval_examples = processor.get_test_examples_file(testF)
        evaluation_data_loader = load_data(eval_examples, label_list, self.tokenizer, seq_length, 2 * batch_size,
                                           DataType.EVALUATION, self.model_type)

        exp_name = 'models/' + dataset_name
        evaluation = Evaluation(evaluation_data_loader, exp_name, exp_name, len(label_list), self.model_type)
        result = evaluation.evaluate(self.model, device, -1, silent)

        try:
            l0 = result.split('\n')[2].split('       ')[2].split('      ')
            l1 = result.split('\n')[3].split('       ')[2].split('      ')
        except:
            l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
            l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')

        p = float(l1[0])
        r = float(l1[1])
        f1 = float(l1[2])
        pnm = float(l0[0])
        rnm = float(l0[1])
        f1nm = float(l0[2])
        return p, r, f1, pnm, rnm, f1nm

    def predict(self, t1, t2, **kwargs):
        x = pd.DataFrame([0] + t1 + t2).T

        device, n_gpu = initialize_gpu_seed(-1)
        processor = DeepMatcherProcessor()
        tmpf = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
        x.to_csv(tmpf)
        examples = processor.get_test_examples_file(tmpf)
        test_data_loader = load_data(examples, processor.get_labels(),
                                     self.tokenizer,
                                     MAX_SEQ_LENGTH,
                                     BATCH_SIZE,
                                     DataType.TEST, self.model_type)

        _, _, _, predictions = prediction.predict(self.model, device, test_data_loader, True, **kwargs)
        os.remove(tmpf)
        return predictions

    def noise(self, tupla, mask='[MASK]'):
        copy_tup = []
        for i in range(len(tupla)):
            change_attr = random.randint(0, 2)
            if len(tupla[i])>1 and change_attr == 1:
                text = str(tupla[i])
                masked_text = text.replace(random.choice(text.split(' ')), mask, 1)
                sequences = self.noise_pipeline(masked_text)
                noised = sequences[random.randint(0, len(sequences) - 1)]['sequence']
                copy_tup.append(noised)
            else:
                copy_tup.append(tupla[i])
        return copy_tup

    def __apply_dropout(self, m):
        if type(m) == nn.Dropout:
            m.train()

    def enable_mcd(self):
        self.model.eval()
        self.model.apply(self.__apply_dropout)


