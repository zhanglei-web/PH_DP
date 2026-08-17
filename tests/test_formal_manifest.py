from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py

from mujoco_shared_control.collection import AutomaticCollector, CollectionConfig
from mujoco_shared_control.collection.datasets import (
    ManifestActorDataset,
    ManifestCriticDataset,
)
from mujoco_shared_control.collection.manifest import (
    FORMAL_CODE_VERSION,
    FORMAL_CONFIG_HASH,
    FORMAL_CONFIG_VERSION,
    FORMAL_SCHEMA_VERSION,
    build_formal_manifest,
    load_manifest,
)


def _make_formal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    collector = AutomaticCollector(CollectionConfig(dataset_root=str(source_root)),
                                   run_id="fixture_source")
    try:
        result = collector.collect_episode(worker_episode_index=0, environment_seed=123)
    finally:
        collector.close()
    root = tmp_path / "dataset"
    specifications = (
        ("nominal_success", "nominal", "success", "task_success"),
        ("normal_recovered", "perturbed", "recovered", "task_success"),
        ("delayed_recovery", "perturbed", "recovered", "delayed_recovery"),
        ("failure", "perturbed", "failure", "settling_timeout"),
    )
    for index, (category, variant, outcome, reason) in enumerate(specifications):
        destination = root / outcome / f"episode_strict_run_{index}.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.path, destination)
        with h5py.File(destination, "r+") as episode:
            episode.attrs["episode_id"] = f"strict_run_{index}"
            episode.attrs["run_id"] = "strict_run"
            episode.attrs["schema_version"] = FORMAL_SCHEMA_VERSION
            episode.attrs["config_version"] = FORMAL_CONFIG_VERSION
            episode.attrs["config_hash"] = FORMAL_CONFIG_HASH
            episode.attrs["expert_code_version"] = FORMAL_CODE_VERSION
            episode.attrs["collection_variant"] = variant
            episode.attrs["outcome"] = outcome
            episode.attrs["termination_reason"] = reason
            episode.attrs["worker_episode_index"] = index
            episode.attrs["environment_seed"] = 9000 + index
            episode.attrs["policy_seed"] = 9000 + index
            episode.attrs["perturbation_seed"] = 1009003 + index
    unrelated = root / "success" / "episode_preflight.h5"
    shutil.copy2(result.path, unrelated)
    return root, tmp_path / "manifest.json"


def test_manifest_split_is_strict_and_loaders_do_not_discover_unlisted_files(
    tmp_path: Path,
) -> None:
    root, output = _make_formal_fixture(tmp_path)
    expected = {name: 1 for name in
                ("nominal_success", "normal_recovered", "delayed_recovery", "failure")}
    manifest = build_formal_manifest(
        root, output, run_id="strict_run", split_seed=7,
        validation_fraction=.5, expected_categories=expected,
    )
    assert manifest["episode_count"] == 4
    assert all("preflight" not in entry["path"] for entry in manifest["episodes"])
    assert load_manifest(output)["content_sha256"] == manifest["content_sha256"]
    actor = ManifestActorDataset(output, "validation")
    critic = ManifestCriticDataset(output, "validation")
    assert len(actor.entries) == 1
    assert len(critic.entries) == 4
    assert all("strict_run" in entry.path.name for entry in critic.entries)


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    root, output = _make_formal_fixture(tmp_path)
    expected = {name: 1 for name in
                ("nominal_success", "normal_recovered", "delayed_recovery", "failure")}
    build_formal_manifest(root, output, run_id="strict_run", split_seed=7,
                          validation_fraction=.5, expected_categories=expected)
    document = json.loads(output.read_text())
    document["run_id"] = "preflight"
    output.write_text(json.dumps(document), encoding="utf-8")
    try:
        load_manifest(output)
    except ValueError as error:
        assert "content hash mismatch" in str(error)
    else:
        raise AssertionError("tampered manifest was accepted")
