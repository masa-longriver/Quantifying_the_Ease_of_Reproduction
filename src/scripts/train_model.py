import argparse
import datetime as dt
import os
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models import Trainer  # noqa: E402
from src.utils import load_config, setup_logger  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument(
    "--result_dir_name", type=str,
    default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

logger = setup_logger()


def main(args: argparse.Namespace):
    logger.info("Start training the diffusion model")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Result directory name: {args.result_dir_name}")

    config = load_config(args.dataset)

    trainer = Trainer(config, args.result_dir_name)
    trainer.train()
    logger.info("Training is finished.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
