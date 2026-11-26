"""Reproducible CUB-200 launcher for the unified VSMR method."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_DATA_ROOT", "/data/lx/fscil/data")
))
INDEX_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_INDEX_LIST_ROOT", os.path.join(DATA_ROOT, "index_list"))
))

ARGS = [
    "--method", "continual_vsmr",
    "-project", "base",
    "-dataset", "cub200",
    "-vit",
    "-dataroot", DATA_ROOT,
    "--index_list_root", INDEX_ROOT,
    "-gpu", "0",
    "-seed", "1",
    "-epochs_base", "50",
    "-epochs_new", "10",
    "-num_workers", "4",
    "-test_batch_size", "64",
    "--continual_classes_per_batch", "16",
    "--continual_samples_per_class", "4",
    "--continual_feature_batch_size", "128",
    "--continual_eval_batch_size", "128",
    "--continual_history_rep_weight", "0.1",
    "--continual_history_proto_weight", "0.1",
    "-out", "VSMR",
]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd=str(ROOT),
        check=True,
    )
