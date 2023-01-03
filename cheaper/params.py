class CheapERParams:

    def __init__(self, fast: bool = False):
        self.epsilon = 0
        self.adaptive_ft = True
        self.normalize = True
        self.sim_length = 5
        self.warmup = True
        self.silent = True
        self.weight_decay = 0.01
        self.lr = 1e-5
        self.lr_multiplier = 3
        self.batch_size = 16
        self.num_runs = 1
        if fast:
            self.compare = False
            self.slicing = [0.05, 0.1, 0.33]
            self.models = ['distilbert-base-uncased']
            self.mask_token = '[MASK]'
            self.epochs = 15
            self.teaching_iterations = 3
            self.sigma = 100
            self.kappa = 10
        else:
            self.slicing = [0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.7, 1]
            self.models = ['roberta-base']
            self.compare = True
            self.mask_token = '<mask>'
            self.teaching_iterations = 5
            self.epochs = 40
            self.sigma = 1000
            self.kappa = 100
        self.attribute_shuffle = False
        self.identity = False
        self.symmetry = False
        self.adjust_ds_size = False
        self.approx = 'perceptron'
        self.generated_only = True
        self.balance = [0.5, 0.5]
        self.deeper_trick = True
        self.consistency = True
        self.sim_edges = True
        self.simple_slicing = True
        self.model_type = 'noisy-student'
        self.data_noise = True
        self.model_noise = True
        self.temperature = 'asc'
        self.discard_old_data = True
        self.use_scores = False
        self.threshold = 0
        self.label_smoothing = 0.1
        self.hf_training = True
        self.seq_length = 0
        self.best_model = 'eval_f1'
        self.mcd_samples = 5

    def __str__(self):
        return 'sigma=' + str(self.sigma) + ',kappa=' + str(self.kappa) + ',epsilon=' + str(
            self.epsilon) + ',adaptive_ft=' \
               + str(self.adaptive_ft) + ',num_runs=' + str(self.num_runs) + ',normalize=' + str(self.normalize) \
               + ',sim_length=' + str(self.sim_length) + ',warmup=' + str(self.warmup) + ',epochs=' + str(self.epochs) \
               + ',lr=' + str(self.lr) + ',attribute_shuffle=' + str(self.attribute_shuffle) + ',identity=' \
               + str(self.identity) + ',symmetry=' + str(self.symmetry) + ',models=' + str(self.models) + ',slicing=' \
               + str(self.slicing) + ',compare=' + str(self.compare) + ',generated_only=' + str(self.generated_only) \
               + ',approx=' + str(self.approx) + ',balance=' + str(self.balance) + ',adjust_ds_size=' \
               + str(self.adjust_ds_size) + ',batch_size=' + str(self.batch_size) + ',silent=' + str(self.silent) \
               + ',deeper_trick=' + str(self.deeper_trick) + ',consistency=' + str(self.consistency) + ',sim_edges=' \
               + str(self.sim_edges) + ',simple_slicing=' + str(self.simple_slicing) + ',use_model=' \
               + str(self.model_type) + ',teaching_iterations=' + str(self.teaching_iterations) + ',lr_multiplier=' + \
               str(self.lr_multiplier) + ',data_noise=' + str(self.data_noise) + ',temperature=' \
               + str(self.temperature) + ',discard_old_data=' + str(self.discard_old_data) + ',use_scores=' + \
               str(self.use_scores) + ',threshold=' + str(self.threshold) + ',weight_decay=' + str(self.weight_decay) + \
               ',label_smoothing=' + str(self.label_smoothing) + ',hf_training=' + str(self.hf_training) +\
               ',seq_length=' + str(self.seq_length) + ',best_model=' + str(self.best_model) + ',mask_token=' \
               + str(self.mask_token) + ',model_noise=' + str(self.model_noise) + ',mcd_samples=' + str(self.mcd_samples)
