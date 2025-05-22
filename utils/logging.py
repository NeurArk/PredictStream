import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger if not already configured."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)

