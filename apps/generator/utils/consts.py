import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USER = os.getenv("USER")

SHARED_DIR_PATH = Path(f"/home/{USER}/app/apps/apps/generator/data/shared_dir")

SPARK_CHECKPOINT_PATH = SHARED_DIR_PATH / "tmp" / "checkpoints"

# /home/sparkuser/app/apps/apps/generator/data/shared_dir/tmp/checkpoints
