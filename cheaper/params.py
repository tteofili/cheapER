from cheaper.emt import config


class CheapERParams:

    def __init__(self):
        self.sigma = 100
        self.kappa = 10
        self.epsilon = 0.015
        self.slicing = [0.1, 0.33, 0.5, 1]
        self.adaptive_ft = True
        self.num_runs = 1
        self.compare = False
        self.normalize = True
        self.sim_length = 5
        self.warmup = True
        self.epochs = 15
        self.lr = 1e-7
        self.lr_multiplier = 200
        self.models = ['roberta-base']
        self.attribute_shuffle = False
        self.identity = False
        self.symmetry = False
        self.adjust_ds_size = False
        self.approx = 'perceptron'
        self.generated_only = True
        self.silent = True
        self.batch_size = 8
        self.balance = [0.5, 0.5]
        self.deeper_trick = True
        self.consistency = True
        self.sim_edges = True
        self.simple_slicing = True
        self.model_type = 'noisy-student'
        self.teaching_iterations = 3
        self.data_noise = True
        self.temperature = None
        self.discard_old_data = False
        self.use_scores = False
        self.threshold = 0

    def __str__(self):
        return 'sigma=' + str(self.sigma) + ',kappa=' + str(self.kappa) + ',epsilon=' + str(self.epsilon) + ',adaptive_ft=' \
               + str(self.adaptive_ft) + ',num_runs=' + str(self.num_runs) + ',normalize=' + str(self.normalize) \
               + ',sim_length=' + str(self.sim_length) + ',warmup=' + str(self.warmup) + ',epochs=' + str(self.epochs) \
               + ',lr=' + str(self.lr) + ',attribute_shuffle=' + str(self.attribute_shuffle) + ',identity=' \
               + str(self.identity) + ',symmetry=' + str(self.symmetry) + ',models=' + str(self.models) + ',slicing=' \
               + str(self.slicing) + ',compare=' + str(self.compare) + ',generated_only=' + str(self.generated_only) \
               + ',approx=' + str(self.approx) + ',balance=' + str(self.balance) + ',adjust_ds_size=' \
               + str(self.adjust_ds_size) + ',batch_size=' + str(self.batch_size) + ',silent=' + str(self.silent) \
               + ',deeper_trick=' + str(self.deeper_trick) + ',consistency=' + str(self.consistency) + ',sim_edges=' \
               + str(self.sim_edges) + ',simple_slicing=' + str(self.simple_slicing) + ',use_model=' \
               + str(self.model_type) + ',teaching_iterations=' + str(self.teaching_iterations) + ',lr_multiplier=' +\
               str(self.lr_multiplier) + ',data_noise=' + str(self.data_noise) + ',temperature=' \
               + str(self.temperature) + ',discard_old_data=' + str(self.discard_old_data) + ',use_scores=' +\
               str(self.use_scores) + ',threshold=' + str(self.threshold)
