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
