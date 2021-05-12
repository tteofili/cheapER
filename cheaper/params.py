from cheaper.emt import config


class CheapERParams:

    def __init__(self):
        self.sigma = 0
        self.kappa = 0
        self.epsilon = 0
        self.slicing = []
        self.pretrain = False
        self.num_runs = 1
        self.compare = False
        self.normalize = True
        self.sim_length = 10
        self.warmup = False
        self.epochs = 3
        self.lr = 1e-3
        self.models = config.Config.MODEL_CLASSES
        self.attribute_shuffle = False
        self.identity = False
        self.symmetry = False

    def __str__(self):
        return 'sigma='+ str(self.sigma) + ',kappa=' + str(self.kappa) + ',epsilon=' + str(self.epsilon) + ',pretrain='\
               + str(self.pretrain) + ',num_runs=' + str(self.num_runs) + ',normalize=' + str(self.normalize) \
               + ',sim_length=' + str(self.sim_length) + ',warmup=' + str(self.warmup) + ',epochs=' + str(self.epochs) \
               + ',lr=' + str(self.lr) + ',attribute_shuffle=' + str(self.attribute_shuffle) + ',identity=' \
               + str(self.identity) + ',symmetry=' + str(self.symmetry) + ',models=' + str(self.models) + ',slicing=' \
               + str(self.slicing) + ',compare=' + str(self.compare)