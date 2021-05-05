import os
import logging
from cheaper.data import deepmatcher_format
from cheaper.emt.config import Config
from cheaper.emt.data_loader import load_data, DataType
from cheaper.emt.data_representation import DeepMatcherProcessor
from cheaper.emt.evaluation import Evaluation
from cheaper.emt.model import save_model, load_model
from cheaper.emt.optimizer import build_optimizer
from cheaper.emt.torch_initializer import initialize_gpu_seed
from cheaper.emt.training import train
from cheaper.emt.logging_customized import setup_logging

setup_logging()

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset

BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128


class EMTERModel:

    def __init__(self, model_type):
        self.model_type = model_type
        config_class, model_class, tokenizer_class, mlm_model_class = Config().MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(self.model_type)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_type, do_lower_case=True)
        self.model = model_class.from_pretrained(self.model_type, config=config)
        self.mlm_model = mlm_model_class.from_pretrained(self.model_type, config=config)

    def pretrain(self, unlabelled_train_file, unlabelled_valid_file, dataset_name, model_type, seq_length=MAX_SEQ_LENGTH, warmup=False,
                 epochs=3, lr=1e-3):

        model_dir = 'models/' + dataset_name + "/mlm-" + model_type
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if os.path.exists(model_dir + '/pytorch_model.bin'):
            self.model = load_model(model_dir)
        else:
            train_dataset = LineByLineTextDataset(
                tokenizer=self.tokenizer,
                file_path=unlabelled_train_file,
                block_size=128,
            )

            valid_dataset = LineByLineTextDataset(
                tokenizer=self.tokenizer,
                file_path=unlabelled_valid_file,
                block_size=128,
            )

            training_args = TrainingArguments(
                output_dir='./results',  # output directory
                num_train_epochs=3,  # total # of training epochs
                per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
                per_device_eval_batch_size=BATCH_SIZE * 4,  # batch size for evaluation
                logging_dir='./logs',  # directory for storing logs
                save_total_limit=2,
                do_eval=True
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
              epochs=3, lr=1e-5, pretrain=False):
        device, n_gpu = initialize_gpu_seed(22)

        if pretrain:
            pt_model_dir = 'models/' + dataset_name + "/mlm-" + model_type
            logging.info('loading pretrained model from {}'.format(pt_model_dir))
            self.model = load_model(pt_model_dir)

        self.model = self.model.to(device)

        processor = DeepMatcherProcessor()
        trainF, validF = deepmatcher_format.tofiles(label_train, label_valid, dataset_name)
        train_examples = processor.get_train_examples_file(trainF)
        label_list = processor.get_labels()
        training_data_loader = load_data(train_examples, label_list, self.tokenizer, seq_length, BATCH_SIZE, DataType.TRAINING,
                                         self.model_type)

        num_epochs = epochs
        num_train_steps = len(training_data_loader) * num_epochs

        learning_rate = lr
        adam_eps = 1e-8
        if warmup:
            warmup_steps = 500
            weight_decay = 0.01
        else:
            warmup_steps = 0
            weight_decay = 0
        optimizer, scheduler = build_optimizer(self.model, num_train_steps, learning_rate, adam_eps, warmup_steps,
                                               weight_decay)

        eval_examples = processor.get_test_examples_file(validF)
        evaluation_data_loader = load_data(eval_examples, label_list, self.tokenizer, seq_length, 4*BATCH_SIZE,
                                           DataType.EVALUATION, self.model_type)

        exp_name = 'models/' + dataset_name
        evaluation = Evaluation(evaluation_data_loader, exp_name, exp_name, len(label_list), self.model_type)

        self.model, result = train(device, training_data_loader, self.model, optimizer, scheduler, evaluation, num_epochs, 1.0,
                       False, experiment_name=exp_name, output_dir=exp_name, model_type=self.model_type)

        save_model(self.model, exp_name, exp_name, tokenizer=self.tokenizer)

        l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
        l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
        p = l1[0]
        r = l1[1]
        f1 = l1[2]
        pnm = l0[0]
        rnm = l0[1]
        f1nm = l0[2]
        return p, r, f1, pnm, rnm, f1nm

    def load(self, path):
        self.model = load_model(path, do_lower_case=True)
        return self

    def save(self, path):
        save_model(self.model, path, path, tokenizer=self.tokenizer)

    def eval(self, label_test, dataset_name, seq_length=MAX_SEQ_LENGTH):
        device, n_gpu = initialize_gpu_seed(22)

        self.model = self.model.to(device)

        processor = DeepMatcherProcessor()
        trainF, testF = deepmatcher_format.tofiles(label_test, label_test, dataset_name)
        label_list = processor.get_labels()

        eval_examples = processor.get_test_examples_file(testF)
        evaluation_data_loader = load_data(eval_examples, label_list, self.tokenizer, seq_length, 4*BATCH_SIZE,
                                           DataType.EVALUATION, self.model_type)

        exp_name = 'models/' + dataset_name
        evaluation = Evaluation(evaluation_data_loader, exp_name, exp_name, len(label_list), self.model_type)
        result = evaluation.evaluate(self.model, device, -1)

        l0 = result['report'].split('\n')[2].split('       ')[2].split('      ')
        l1 = result['report'].split('\n')[3].split('       ')[2].split('      ')
        p = l1[0]
        r = l1[1]
        f1 = l1[2]
        pnm = l0[0]
        rnm = l0[1]
        f1nm = l0[2]
        return p, r, f1, pnm, rnm, f1nm
