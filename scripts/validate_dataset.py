from __future__ import annotations

import argparse
import json
from pathlib import Path

from mujoco_shared_control.data.recording import validate_episode


def _episode_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob("episode_*.h5")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate synchronized MuJoCo HDF5 episodes."
    )
    parser.add_argument("paths", nargs="+", help="Episode files or dataset folders")
    args = parser.parse_args()

    paths = _episode_paths(args.paths)
    if not paths:
        raise SystemExit("No episode HDF5 files found")
    invalid = 0
    for path in paths:
        report = validate_episode(path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        invalid += int(not report["valid"])
    raise SystemExit(1 if invalid else 0)


if __name__ == "__main__":
    main()
