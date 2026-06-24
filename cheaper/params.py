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
        self.slicing = [0.05, 0.1, 0.33, 0.5, 0.7, 1]
        self.compare = False
        if fast:
            self.models = ['distilroberta-base']
            self.epochs = 20
            self.teaching_iterations = 5
        else:
            self.models = ['roberta-base']
            self.teaching_iterations = 7
            self.epochs = 40
        self.mcd_samples = 1
        self.mask_token = '<mask>'
        self.sigma = 1000 # numero di tuple per addestrare lo student
        self.kappa = 100
        self.attribute_shuffle = False
        self.identity = False
        self.symmetry = False
        self.adjust_ds_size = True
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
        self.layers_increase = 0
        self.best_model = 'eval_f1'
        self.sample_tag = False # refer to [Tagged Back-Translation](https://arxiv.org/abs/1906.06442)
        # Battleship-style initial labeling (when training_data_source == 'battleship_oracle')
        self.training_data_source = 'battleship_oracle'  # or 'ground_truth'
        self.initial_budget = 100
        self.oracle_type = 'llm'  # 'similarity' | 'sbert' | 'llm'
        self.first_iter_strategy = 'centrality'  # 'random' | 'diversity' | 'centrality'
        self.unlabeled_train_csv = None  # path to unlabeled candidate pairs CSV (id1, id2); if None, derived from dataset
        # Centrality strategy (when first_iter_strategy == 'centrality')
        self.centrality_criterion = 'pagerank'  # 'pagerank' | 'bc' (betweenness)
        self.centrality_nn_param = 16  # k-NN per node when building graph
        # float (e.g. 0.5) or 'median' | 'otsu' | 'p75' for data-driven threshold (similarity/SBERT only)
        self.oracle_threshold = 'otsu'
        self.oracle_sbert_model = 'all-MiniLM-L6-v2'
        self.first_iter_seed = 42

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
               + str(self.mask_token) + ',model_noise=' + str(self.model_noise) + ',mcd_samples=' + str(self.mcd_samples)\
               + ', sample_tag=' + str(self.sample_tag) + ', layers_increase=' + str(self.layers_increase) \
               + ', training_data_source=' + str(self.training_data_source) + ', initial_budget=' + str(self.initial_budget) \
               + ', oracle_type=' + str(self.oracle_type) + ', first_iter_strategy=' + str(self.first_iter_strategy) \
               + ', oracle_threshold=' + str(self.oracle_threshold) \
               + ', centrality_criterion=' + str(self.centrality_criterion) \
               + ', centrality_nn_param=' + str(self.centrality_nn_param)
