import json
from pathlib import Path


class Config:
    def __init__(self, dataset) -> None:
        # base config
        self.dataset = dataset
        self.reconstruct = False
        self.log_step = 20
        self.gpu_device = '0'

        self.modalities = ['metric', 'trace', 'log']
        
        # alert config
        self.metric_direction = True
        self.trace_op = True
        self.trace_ab_type = True

        # TVDiag modules
        self.aug_percent = 0.2
        self.aug_times = 10
        self.TO = True
        self.CM = True
        self.dynamic_weight = True

        # model config
        self.temperature = 0.3
        self.contrastive_loss_scale = 0.1
        self.batch_size = 128
        self.epochs = 500
        self.alert_embedding_dim = 128
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001

        self.feat_drop = 0.1
        self.patience = 20
        self.ft_num = None
        self.aggregator = 'mean'
        self.failure_types = []

        # A new dataset can override model hyperparameters without changing
        # this class. Label-dependent values such as ft_num are inferred by
        # EventProcess after label.csv is loaded.
        dataset_config_path = Path('data') / dataset / 'dataset_config.json'
        if dataset_config_path.exists():
            with dataset_config_path.open('r', encoding='utf-8') as handle:
                overrides = json.load(handle)
            unknown = sorted(set(overrides) - set(vars(self)))
            if unknown:
                raise ValueError(
                    f"Unsupported keys in {dataset_config_path}: {unknown}"
                )
            for key, value in overrides.items():
                setattr(self, key, value)
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")
