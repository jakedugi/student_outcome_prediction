import logging
import time
from functools import wraps
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("sopipeline")


def timer(fn):
    """Simple decorator to log a function's runtime."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("⏱  %s finished in %.2fs", fn.__name__, elapsed)
        return result

    return wrapper


def ensure_dir(p: Path) -> Path:
    p.mkdir(exist_ok=True, parents=True)
    return p