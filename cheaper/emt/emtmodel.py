from keras.models import Model, load_model
from cheaper.data import deepmatcher_format
from cheaper.emt.config import Config
from cheaper.emt.data_loader import load_data, DataType
from cheaper.emt.data_representation import DeepMatcherProcessor
from cheaper.emt.evaluation import Evaluation
from cheaper.emt.model import save_model
from cheaper.emt.optimizer import build_optimizer
from cheaper.emt.torch_initializer import initialize_gpu_seed
from cheaper.emt.training import train

BATCH_SIZE = 8

MAX_SEQ_LENGTH = 128


class EMTERModel:

    def __init__(self, model_type):
        self.model_type = model_type
        config_class, model_class, tokenizer_class = Config().MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(self.model_type)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_type, do_lower_case=True)
        self.model = model_class.from_pretrained(self.model_type, config=config)

    def train(self, label_train, label_valid, label_test, dataset_name):
        device, n_gpu = initialize_gpu_seed(22)

        self.model = self.model.to(device)

        processor = DeepMatcherProcessor()
        trainF, testF = deepmatcher_format.tofiles(label_train, label_test, dataset_name)
        train_examples = processor.get_train_examples_file(trainF)
        label_list = processor.get_labels()
        training_data_loader = load_data(train_examples, label_list, self.tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE, DataType.TRAINING,
                                         self.model_type)

        num_epochs = 3
        num_train_steps = len(training_data_loader) * num_epochs

        learning_rate = 2e-5
        adam_eps = 1e-8
        warmup_steps = 0
        weight_decay = 0
        optimizer, scheduler = build_optimizer(self.model, num_train_steps, learning_rate, adam_eps, warmup_steps,
                                               weight_decay)

        eval_examples = processor.get_test_examples_file(testF)
        evaluation_data_loader = load_data(eval_examples, label_list, self.tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE,
                                           DataType.EVALUATION, self.model_type)

        exp_name = 'datasets/temporary/' + dataset_name
        evaluation = Evaluation(evaluation_data_loader, exp_name, exp_name, len(label_list), self.model_type)

        result = train(device, training_data_loader, self.model, optimizer, scheduler, evaluation, num_epochs, 1.0,
                       True, experiment_name=exp_name, output_dir=exp_name, model_type=self.model_type)

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
        self.model, self.tokenizer = load_model(path, True)
        return self

    def save(self, path):
        save_model(self.model, path, path, tokenizer=self.tokenizer)
