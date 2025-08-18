import importlib.util
import os

from xarch_tokenizers.logging.logger import setup_basic_logger

logger = setup_basic_logger("utils")


def find_package_dir(package: str) -> str:
    # Find the lm_eval module path:
    spec = importlib.util.find_spec(package)
    pkg_dir = ""
    if spec and spec.origin:
        pkg_dir = os.path.dirname(spec.origin)
    elif spec and spec.submodule_search_locations:
        pkg_dir = os.path.dirname(spec.submodule_search_locations[0])
    else:
        logger.warning("lm_eval not found in Python path")
    return pkg_dir
