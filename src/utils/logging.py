import logging


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with a specified logging level.

    Args:
        level (int): The logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
