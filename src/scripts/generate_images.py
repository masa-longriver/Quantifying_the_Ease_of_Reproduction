import argparse
import datetime as dt
import os
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models import Generator  # noqa: E402
from src.utils import load_config, setup_logger, load_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument(
    "--result_dir", type=str,
    default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

logger = setup_logger()


def main(args: argparse.Namespace) -> None:
    logger.info("Start generating images with the trained model")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Result directory name: {args.result_dir}")

    config = load_config(args.dataset)
    model = load_model(config)

    generator = Generator(config, model, args.result_dir)
    generator.generate()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
