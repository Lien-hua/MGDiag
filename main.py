import argparse
import os
import time
import random
from config.exp_config import Config
from helper.logger import get_logger
import torch
import numpy as np
from core.MGDiag import TVDiag
from helper.Result import Result
from process.EventProcess import EventProcess
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(config: Config, logger):
    reconstruct = config.reconstruct
    logger.info("[Data 1/2] Initialize event processor")
    processor = EventProcess(config, logger)
    logger.info(f"[Data 2/2] Process dataset (reconstruct={reconstruct})")
    train_data, aug_data, test_data = processor.process(reconstruct=reconstruct)
    logger.info(
        "Dataset ready: %d training samples, %d augmented samples, %d test samples",
        len(train_data), len(aug_data), len(test_data)
    )

    return train_data, aug_data, test_data


def train_and_evaluate(config: Config, log_dir, exp_name):
    started_at = time.time()
    set_seed(2)
    logger = get_logger(log_dir, exp_name)
    logger.info("=" * 72)
    logger.info("Start experiment: %s", exp_name)
    logger.info("[Step 1/4] Load and prepare dataset")
    train_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("[Step 2/4] Initialize TVDiag model")
    model = TVDiag(config, logger, log_dir)
    logger.info("[Step 3/4] Train model")
    model.train(train_data, aug_data)
    logger.info("[Step 4/4] Evaluate model")
    res: Result = model.evaluate(test_data)
    result_df = res.export_df(exp_name)
    prediction_path = os.path.join(log_dir, "predictions.csv")
    res.export_predictions().to_csv(prediction_path, index=False)
    logger.info("Experiment completed in %.2f seconds", time.time() - started_at)
    logger.info("Results:\n%s", result_df.to_string(index=False))
    logger.info("Per-case predictions: %s", prediction_path)
    logger.info("=" * 72)
    return result_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MGDiag")
    parser.add_argument("--dataset", default="aiops2025")
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="rebuild FastText event embeddings from raw event JSON files",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--aug-times", type=int)
    parser.add_argument("--log-dir")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config(args.dataset)
    config.reconstruct = args.reconstruct
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.aug_times is not None:
        config.aug_times = args.aug_times
    log_dir = args.log_dir or os.path.join("logs", args.dataset)
    train_and_evaluate(config, log_dir, args.dataset)
