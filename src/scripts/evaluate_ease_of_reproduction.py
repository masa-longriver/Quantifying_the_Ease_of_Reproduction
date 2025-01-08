import argparse
import datetime as dt
import os
import sys

import torch

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.evaluation import Evaluator  # noqa: E402
from src.utils import (  # noqa: E402
    load_config, setup_logger, load_model, ResultWriter
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument(
    "--result_dir_name", type=str,
    default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

logger = setup_logger()


def main(args: argparse.Namespace) -> None:
    logger.info("Start evaluating the ease of reproduction")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Result directory name: {args.result_dir_name}")

    config = load_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    model = load_model(config)
    result_writer = ResultWriter(args.result_dir_name, config)
    evaluator = Evaluator(config, model, result_writer)
    evaluator.evaluate()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
