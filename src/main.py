from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _rewrite_run_override() -> None:
    new_argv = []
    for arg in sys.argv:
        if arg.startswith("run="):
            new_argv.append("runs@run=" + arg.split("=", 1)[1])
        else:
            new_argv.append(arg)
    sys.argv = new_argv


_rewrite_run_override()

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'.")

    repo_root = Path(__file__).resolve().parents[1]
    run_id = cfg.run.run_id

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching subprocess:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
