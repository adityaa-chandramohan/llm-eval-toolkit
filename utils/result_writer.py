import os
import json
import pandas as pd
from datetime import datetime
from config.eval_config import config
from utils.logger import get_logger

logger = get_logger(__name__)


def write_results(suite_name: str, df: pd.DataFrame) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.output_dir, f"{suite_name}_{timestamp}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Results saved → {path}")
    return path


def write_summary(suite_name: str, summary: dict) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.output_dir, f"{suite_name}_summary_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {path}")
    return path
