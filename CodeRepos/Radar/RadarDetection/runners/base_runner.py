import torch


class BaseRunner(object):
    def __init__(self, configs):
        self.configs = configs
        self.config = configs['all_args']
        self.device = configs['device']
        self.run_dir = configs['run_dir']
        self.save_name = configs['save_name']
        self.model_save_path = self.run_dir / self.save_name
        self.model_load_path = self.model_save_path
        self.train = self.config.Train
        self.continue_train = self.config.ContinueTrain

        # training param
        self.channels = self.config.channels
        self.epochs = self.config.epochs
        self.learning_rate = self.config.learning_rate

        # loss function param
        self.ctype = self.config.ctype

        # NOT LOAD PARAM
        self.dataset = None
        self.data_loader = None

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError
