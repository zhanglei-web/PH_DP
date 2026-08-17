from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from mujoco_shared_control.awac.dataset import convert_formal_rule_dataset
from mujoco_shared_control.collection.manifest import (
    MANIFEST_VERSION,
    manifest_content_hash,
    sha256_file,
)


def _write_episode(path: Path, episode_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = 6
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as episode:
        attrs = {
            "episode_id": episode_id,
            "run_id": "awac_fixture",
            "schema_version": "2.0.0",
            "config_version": "rule_collection_v1",
            "config_hash": "fixture_config_hash",
            "expert_code_version": "fixture_code_version",
        }
        for name, value in attrs.items():
            episode.attrs[name] = value

        def write(name: str, values: np.ndarray) -> None:
            group, leaf = name.rsplit("/", 1)
            episode.require_group(group).create_dataset(leaf, data=values)

        obs = np.arange(length * 42, dtype=np.float32).reshape(length, 42)
        action = np.zeros((length, 7), dtype=np.float64)
        action[5, 0] = 1.1
        clipped = np.zeros((length, 7), dtype=np.float64)
        clipped[:, 6] = 0.04
        status = np.asarray(
            [
                [1, 0, 0, 1],  # accepted
                [1, 1, 0, 1],  # accepted after clipping
                [0, 0, 1, 0],  # fallback
                [1, 0, 0, 1],  # invalid expert
                [0, 0, 0, 0],  # rejected without fallback
                [1, 0, 0, 1],  # out-of-range/unconfirmed action
            ],
            dtype=np.uint8,
        )
        rejection = np.asarray(["", "", "ik_nonconvergence", "", "ik_rejected", ""])

        write("observations/policy_state_42", obs)
        write("next_observations/policy_state_42", obs + 1)
        write("actions/normalized", action)
        write("actions/expert_nominal", np.full((length, 7), 0.2))
        write("actions/command_after_clipping", clipped)
        write("actions/executed_joint_target", np.zeros((length, 8)))
        write("actions/mujoco_ctrl", np.zeros((length, 8)))
        write("actions/expert_valid", np.asarray([1, 1, 1, 0, 1, 1], np.uint8))
        write("actions/status", status)
        write("actions/rejection_reason", rejection.astype(string_dtype))
        write("identity/step_index", np.arange(length, dtype=np.int64))
        write("labels/reward", np.arange(length, dtype=np.float64))
        write("labels/terminated", np.asarray([0, 0, 0, 0, 0, 1], np.uint8))
        write("labels/truncated", np.zeros(length, np.uint8))
        write("labels/task_success", np.zeros(length, np.uint8))
        write("labels/termination_reason", np.asarray([""] * length).astype(string_dtype))
        write("labels/expert_stage", np.arange(length, dtype=np.uint8))
        write("labels/stage", np.minimum(np.arange(length), 4).astype(np.uint8))
        write("labels/events", np.arange(length, dtype=np.uint32))
        write("labels/task_milestones", np.zeros((length, 5), np.uint8))
        write("perturbations/active", np.asarray([0, 1, 1, 0, 0, 0], np.uint8))
        write("perturbations/magnitude", np.linspace(0, 1, length))


def _make_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "episodes"
    specifications = (
        ("train_ep", "train", "nominal_success"),
        ("validation_ep", "validation", "failure"),
    )
    entries = []
    for episode_id, split, category in specifications:
        path = root / f"{episode_id}.h5"
        _write_episode(path, episode_id)
        entries.append(
            {
                "episode_id": episode_id,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "transitions": 6,
                "outcome": "success" if category == "nominal_success" else "failure",
                "variant": "nominal" if category == "nominal_success" else "perturbed",
                "category": category,
                "split": split,
            }
        )
    # A malformed, unlisted HDF5 proves conversion does not discover recursively.
    (root / "preflight").mkdir()
    with h5py.File(root / "preflight" / "unlisted.h5", "w"):
        pass
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": "awac_fixture",
        "dataset_root": "episodes",
        "dataset_root_base": "manifest_parent",
        "run_id": "awac_fixture",
        "schema_version": "2.0.0",
        "config_version": "rule_collection_v1",
        "config_hash": "fixture_config_hash",
        "code_version": "fixture_code_version",
        "episode_count": 2,
        "transition_count": 12,
        "episodes": entries,
    }
    manifest["content_sha256"] = manifest_content_hash(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_awac_conversion_is_manifest_only_and_filters_invalid_actions(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest(tmp_path)
    output = tmp_path / "awac"
    report = convert_formal_rule_dataset(manifest, output)

    assert report["manifest_episode_count"] == 2
    assert report["episodes_by_split"] == {"train": 1, "validation": 1}
    assert report["raw_transition_count"] == 12
    assert report["kept_transition_count"] == 4
    assert report["exclusions_exclusive"] == {
        "ik_rejection": 2,
        "fallback": 2,
        "invalid_expert": 2,
        "unconfirmed_action_execution": 2,
    }
    assert report["source_flags_nonexclusive"]["action_clipped"] == 2
    assert report["action_validation"]["all_within_closed_interval_minus1_plus1"]
    assert report["observation_validation"]["all_obs_and_next_obs_are_42d"]
    assert report["nonfinite"]["final_network_value_count"] == 0

    with np.load(output / "train.npz", allow_pickle=False) as dataset:
        assert dataset["obs"].shape == (2, 42)
        assert dataset["next_obs"].shape == (2, 42)
        assert dataset["action"].shape == (2, 7)
        assert dataset["action"].dtype == np.float32
        assert dataset["reward"].dtype == np.float32
        assert dataset["terminated"].dtype == np.bool_
        assert dataset["truncated"].dtype == np.bool_
        np.testing.assert_array_equal(dataset["step_index"], [0, 1])
        assert np.all(dataset["action"] == 0.0)
        assert np.all(dataset["expert_nominal"] == np.float32(0.2))
        assert bool(dataset["status"][1, 1])


def test_awac_conversion_rejects_a_manifest_checksum_mismatch(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    source = tmp_path / "episodes" / document["episodes"][0]["path"]
    copy = tmp_path / "tampered.h5"
    shutil.copy2(source, copy)
    with h5py.File(source, "r+") as episode:
        episode["labels/reward"][0] = 999.0
    try:
        convert_formal_rule_dataset(manifest, tmp_path / "output")
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("tampered formal HDF5 was accepted")
