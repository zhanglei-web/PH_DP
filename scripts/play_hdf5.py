#!/usr/bin/env python3
"""Interactively replay one synchronized RGB-D HDF5 episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageTk


STAGE_NAMES = {
    0: "APPROACH / 接近",
    1: "GRASP / 抓取",
    2: "TRANSPORT / 搬运",
    3: "PLACE / 放置",
    4: "COMPLETE / 完成",
}

EVENT_NAMES = (
    (1 << 0, "GRIP_PRESSED"),
    (1 << 1, "GRIP_RELEASED"),
    (1 << 2, "GRASP_ACQUIRED"),
    (1 << 3, "GRASP_LOST"),
    (1 << 4, "ENTERED_GOAL"),
    (1 << 5, "TASK_SUCCESS"),
)

ACTION_STATUS_NAMES = (
    "ik_success",
    "command_accepted",
    "action_clipped",
    "fallback_used",
)


def _resolve_episode(value: str | None) -> Path:
    """Resolve a file, or select the newest finalized episode in a directory."""
    path = Path(value).expanduser() if value else Path("datasets")
    if path.is_file():
        return path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    candidates = [
        candidate
        for candidate in path.rglob("*.h5")
        if ".inprogress." not in candidate.name
    ]
    if not candidates:
        raise FileNotFoundError(f"目录中没有已保存的 HDF5 Episode: {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime).resolve()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _format_array(value: Any, precision: int = 4) -> str:
    array = np.asarray(value)
    return np.array2string(
        array,
        precision=precision,
        suppress_small=True,
        separator=", ",
        max_line_width=120,
    )


def _event_text(mask: int) -> str:
    names = [name for bit, name in EVENT_NAMES if mask & bit]
    return " | ".join(names) if names else "NONE"


def _depth_to_rgb(depth: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Convert metric depth to a dependency-free, high-contrast color image."""
    finite = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[finite] = np.clip(
        (depth[finite] - minimum) / (maximum - minimum), 0.0, 1.0
    )

    # Near to far: red -> yellow -> cyan -> blue. Invalid pixels remain black.
    positions = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    colors = np.array(
        [[255, 48, 32], [255, 230, 32], [32, 220, 220], [35, 45, 180]],
        dtype=np.float32,
    )
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.interp(
            normalized, positions, colors[:, channel]
        ).astype(np.uint8)
    output[~finite] = 0
    return output


class Episode:
    """Thin, lazy reader that keeps image frames on disk until requested."""

    def __init__(self, path: Path, camera: str | None = None) -> None:
        self.path = path
        self.file = h5py.File(path, "r")
        self.camera_names = self._camera_names()
        if not self.camera_names:
            self.close()
            raise ValueError("HDF5 中没有成对的 RGB 和深度图像")
        self.camera = camera or self.camera_names[0]
        if self.camera not in self.camera_names:
            available = ", ".join(self.camera_names)
            self.close()
            raise ValueError(f"相机 {self.camera!r} 不存在，可选: {available}")

        self.rgb_path = f"observations/images/{self.camera}/rgb"
        self.depth_path = f"observations/images/{self.camera}/depth"
        self.length = int(self.file[self.rgb_path].shape[0])
        if self.length == 0:
            self.close()
            raise ValueError("Episode 不包含任何帧")
        if self.file[self.depth_path].shape[0] != self.length:
            self.close()
            raise ValueError("RGB 与深度帧数不一致")
        self.sample_rate = float(self.file.attrs.get("sample_rate_hz", 20.0))
        raw_names = self.file.attrs.get("state_26_names_json", "[]")
        try:
            self.state_names = json.loads(_decode(raw_names))
        except json.JSONDecodeError:
            self.state_names = []

    def _camera_names(self) -> list[str]:
        root = self.file.get("observations/images")
        if not isinstance(root, h5py.Group):
            return []
        return sorted(
            name
            for name, group in root.items()
            if isinstance(group, h5py.Group) and "rgb" in group and "depth" in group
        )

    def close(self) -> None:
        if self.file:
            self.file.close()

    def value(self, name: str, index: int, default: Any = None) -> Any:
        dataset = self.file.get(name)
        if not isinstance(dataset, h5py.Dataset) or index >= dataset.shape[0]:
            return default
        return dataset[index]

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.file[self.rgb_path][index], self.file[self.depth_path][index]

    def timestamp(self, index: int) -> float:
        value = self.value("timestamps/simulation", index)
        if value is None:
            return index / self.sample_rate
        return float(value)


class EpisodePlayer:
    def __init__(
        self,
        episode: Episode,
        *,
        speed: float,
        paused: bool,
        depth_min: float,
        depth_max: float,
        display_width: int,
    ) -> None:
        self.episode = episode
        self.speed = speed
        self.playing = not paused
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.display_width = display_width
        self.index = 0
        self._after_id: str | None = None
        self._updating_slider = False
        self._rgb_photo: ImageTk.PhotoImage | None = None
        self._depth_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title(f"HDF5 Episode 回放 - {episode.path.name}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", self._on_key)
        self._build_ui()
        self._show_frame()
        self._schedule_next()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.Frame(outer)
        image_frame.grid(row=0, column=0, sticky="nsew")
        info_frame = ttk.Frame(outer)
        info_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        self.rgb_title = ttk.Label(image_frame, text="RGB", anchor=tk.CENTER)
        self.rgb_title.pack(fill=tk.X)
        self.rgb_label = ttk.Label(image_frame)
        self.rgb_label.pack(fill=tk.BOTH, expand=True)
        self.depth_title = ttk.Label(image_frame, text="Depth", anchor=tk.CENTER)
        self.depth_title.pack(fill=tk.X, pady=(8, 0))
        self.depth_label = ttk.Label(image_frame)
        self.depth_label.pack(fill=tk.BOTH, expand=True)

        self.info = tk.Text(
            info_frame,
            width=72,
            height=42,
            wrap=tk.NONE,
            font=("TkFixedFont", 10),
            padx=8,
            pady=8,
        )
        info_scroll_y = ttk.Scrollbar(
            info_frame, orient=tk.VERTICAL, command=self.info.yview
        )
        info_scroll_x = ttk.Scrollbar(
            info_frame, orient=tk.HORIZONTAL, command=self.info.xview
        )
        self.info.configure(
            yscrollcommand=info_scroll_y.set, xscrollcommand=info_scroll_x.set
        )
        self.info.grid(row=0, column=0, sticky="nsew")
        info_scroll_y.grid(row=0, column=1, sticky="ns")
        info_scroll_x.grid(row=1, column=0, sticky="ew")
        info_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)

        controls = ttk.Frame(outer, padding=(0, 8, 0, 0))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)
        self.play_button = ttk.Button(controls, command=self._toggle_play)
        self.play_button.grid(row=0, column=0, padx=(0, 8))
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=max(0, self.episode.length - 1),
            command=self._on_seek,
        )
        self.slider.grid(row=0, column=1, sticky="ew")
        self.status = ttk.Label(controls, width=38, anchor=tk.E)
        self.status.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(
            controls,
            text=(
                "空格: 播放/暂停   ←/→: 逐帧   ↑/↓: 调速   "
                "Home/End: 首/尾帧   Q/Esc: 退出"
            ),
            anchor=tk.CENTER,
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def run(self) -> None:
        self.root.mainloop()

    def close(self, *_: Any) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.episode.close()
        self.root.destroy()

    def _schedule_next(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if not self.playing:
            return
        if self.index + 1 < self.episode.length:
            delta = self.episode.timestamp(self.index + 1) - self.episode.timestamp(
                self.index
            )
        else:
            delta = 1.0 / self.episode.sample_rate
        if not np.isfinite(delta) or delta <= 0.0 or delta > 1.0:
            delta = 1.0 / self.episode.sample_rate
        delay_ms = max(1, round(1000.0 * delta / self.speed))
        self._after_id = self.root.after(delay_ms, self._advance)

    def _advance(self) -> None:
        self._after_id = None
        if not self.playing:
            return
        if self.index >= self.episode.length - 1:
            self.playing = False
        else:
            self.index += 1
            self._show_frame()
        self._schedule_next()

    def _toggle_play(self) -> None:
        if self.index >= self.episode.length - 1 and not self.playing:
            self.index = 0
            self._show_frame()
        self.playing = not self.playing
        self._update_controls()
        self._schedule_next()

    def _seek(self, index: int, *, pause: bool = True) -> None:
        self.index = min(max(index, 0), self.episode.length - 1)
        if pause:
            self.playing = False
        self._show_frame()
        self._schedule_next()

    def _on_seek(self, value: str) -> None:
        if self._updating_slider:
            return
        self._seek(round(float(value)))

    def _on_key(self, event: tk.Event[Any]) -> None:
        key = event.keysym.lower()
        if key == "space":
            self._toggle_play()
        elif key == "left":
            self._seek(self.index - 1)
        elif key == "right":
            self._seek(self.index + 1)
        elif key == "home":
            self._seek(0)
        elif key == "end":
            self._seek(self.episode.length - 1)
        elif key == "up":
            self.speed = min(8.0, self.speed * 2.0)
            self._update_controls()
            self._schedule_next()
        elif key == "down":
            self.speed = max(0.25, self.speed / 2.0)
            self._update_controls()
            self._schedule_next()
        elif key in {"q", "escape"}:
            self.close()

    def _show_frame(self) -> None:
        rgb, depth = self.episode.frame(self.index)
        rgb_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        depth_image = Image.fromarray(
            _depth_to_rgb(depth, self.depth_min, self.depth_max), mode="RGB"
        )
        target_height = max(1, round(rgb_image.height * self.display_width / rgb_image.width))
        size = (self.display_width, target_height)
        rgb_image = rgb_image.resize(size, Image.Resampling.BILINEAR)
        depth_image = depth_image.resize(size, Image.Resampling.NEAREST)
        self._rgb_photo = ImageTk.PhotoImage(rgb_image)
        self._depth_photo = ImageTk.PhotoImage(depth_image)
        self.rgb_label.configure(image=self._rgb_photo)
        self.depth_label.configure(image=self._depth_photo)
        self.depth_title.configure(
            text=f"Depth（红色近、蓝色远，显示范围 {self.depth_min:.2f}–{self.depth_max:.2f} m）"
        )
        self._update_info(depth)
        self._update_controls()

    def _update_controls(self) -> None:
        self.play_button.configure(text="暂停" if self.playing else "播放")
        self._updating_slider = True
        self.slider.set(self.index)
        self._updating_slider = False
        self.status.configure(
            text=(
                f"帧 {self.index + 1}/{self.episode.length}   "
                f"{self.speed:g}×   {'播放中' if self.playing else '已暂停'}"
            )
        )

    def _update_info(self, depth: np.ndarray) -> None:
        get = self.episode.value
        i = self.index
        timestamp = self.episode.timestamp(i)
        step = int(get("identity/step_index", i, i))
        stage = int(get("labels/stage", i, 0))
        events = int(get("labels/events", i, 0))
        image_valid = bool(get(f"camera/{self.episode.camera}/image_valid", i, 1))
        status = np.asarray(get("actions/status", i, np.zeros(4, dtype=np.uint8)))
        status_text = ", ".join(
            f"{name}={bool(value)}"
            for name, value in zip(ACTION_STATUS_NAMES, status, strict=False)
        )
        finite_depth = depth[np.isfinite(depth) & (depth > 0.0)]
        if finite_depth.size:
            depth_stats = (
                f"min/mean/max={finite_depth.min():.3f}/"
                f"{finite_depth.mean():.3f}/{finite_depth.max():.3f} m"
            )
        else:
            depth_stats = "无有效深度"

        state_26 = np.asarray(get("observations/state_26", i, []))
        state_lines = []
        if len(self.episode.state_names) == state_26.size:
            for start in range(0, state_26.size, 2):
                entries = [
                    f"{self.episode.state_names[j]}={state_26[j]: .4f}"
                    for j in range(start, min(start + 2, state_26.size))
                ]
                state_lines.append("  " + "    ".join(entries))
        else:
            state_lines.append("  " + _format_array(state_26))

        rejection = _decode(get("actions/rejection_reason", i, ""))
        drop_reason = _decode(
            get(f"camera/{self.episode.camera}/drop_reason", i, "")
        )
        camera_root = f"camera/{self.episode.camera}"
        image_age = float(get(f"{camera_root}/image_age_ms", i, np.nan))
        sync_error = float(
            get(f"{camera_root}/state_image_sync_error_ms", i, np.nan)
        )
        render_latency = float(
            get(f"{camera_root}/render_latency_ms", i, np.nan)
        )
        object_linear_velocity = _format_array(
            get("observations/object_linear_velocity", i, [])
        )
        object_angular_velocity = _format_array(
            get("observations/object_angular_velocity", i, [])
        )
        contact = _format_array(get("observations/contact", i, []))
        vr_valid = bool(get("actions/vr_raw_valid", i, 0))
        vr_aligned = bool(get("actions/vr_aligned", i, 0))
        vr_age = float(get("actions/vr_age_ms", i, np.nan))
        user_valid = bool(get("actions/user_command_valid", i, 0))
        user_age = float(get("actions/user_command_age_ms", i, np.nan))
        policy_valid = bool(get("actions/policy_output_valid", i, 0))
        policy_confidence = float(
            get("actions/policy_output_confidence", i, 0.0)
        )
        policy_space = int(get("actions/policy_output_command_space", i, 0))
        policy_active = bool(get("actions/policy_output_control_active", i, 0))
        policy_age = float(get("actions/policy_output_age_ms", i, np.nan))
        lines = [
            "EPISODE",
            f"  文件: {self.episode.path}",
            f"  task: {_decode(self.episode.file.attrs.get('task_name', ''))}",
            f"  episode_id: {_decode(self.episode.file.attrs.get('episode_id', ''))}",
            f"  schema: {_decode(self.episode.file.attrs.get('schema_version', ''))}",
            f"  nominal_rate: {self.episode.sample_rate:g} Hz",
            "",
            "CURRENT FRAME",
            f"  step_index={step}  simulation_time={timestamp:.6f} s",
            f"  stage={stage}: {STAGE_NAMES.get(stage, 'UNKNOWN')}",
            f"  events={events}: {_event_text(events)}",
            f"  reward={float(get('labels/reward', i, 0.0)):.5f}",
            f"  task_success={bool(get('labels/task_success', i, 0))}",
            "",
            f"CAMERA [{self.episode.camera}]",
            f"  image_valid={image_valid}  drop_reason={drop_reason or '-'}",
            f"  depth: {depth_stats}",
            f"  image_age={image_age:.3f} ms",
            f"  state/image sync error={sync_error:.3f} ms",
            f"  render latency={render_latency:.3f} ms",
            "",
            "OBSERVATION",
            f"  q: {_format_array(get('observations/joint_position', i, []))}",
            f"  dq: {_format_array(get('observations/joint_velocity', i, []))}",
            f"  ee xyz+wxyz: {_format_array(get('observations/ee_pose_xyz_wxyz', i, []))}",
            f"  gripper_opening: {float(get('observations/gripper_opening', i, np.nan)):.5f} m",
            f"  object xyz+wxyz: {_format_array(get('observations/object_pose_xyz_wxyz', i, []))}",
            f"  goal xyz+wxyz: {_format_array(get('observations/goal_pose_xyz_wxyz', i, []))}",
            f"  object linear velocity: {object_linear_velocity}",
            f"  object angular velocity: {object_angular_velocity}",
            f"  contact [L,R,L_force,R_force,count]: {contact}",
            f"  object_grasped: {bool(get('observations/object_grasped', i, 0))}",
            "",
            "ACTION / COMMAND",
            f"  VR raw [xyz,xyzw,trigger,grip]: {_format_array(get('actions/vr_raw', i, []))}",
            f"  VR valid={vr_valid} aligned={vr_aligned} age={vr_age:.3f} ms",
            "  user command [xyz,wxyz,gripper]: "
            f"{_format_array(get('actions/user_command', i, []))}",
            f"  user command valid={user_valid} age={user_age:.3f} ms",
            f"  policy output: {_format_array(get('actions/policy_output', i, []))}",
            f"  policy valid={policy_valid} confidence={policy_confidence:.3f}",
            f"  policy space={policy_space} active={policy_active} "
            f"age={policy_age:.3f} ms",
            f"  executed [q1..q7,gripper]: {_format_array(get('actions/executed', i, []))}",
            f"  mujoco_ctrl: {_format_array(get('actions/mujoco_ctrl', i, []))}",
            f"  status: {status_text}",
            f"  rejection_reason: {rejection or '-'}",
            "",
            "STATE_26",
            *state_lines,
        ]
        self.info.configure(state=tk.NORMAL)
        current_y = self.info.yview()[0]
        self.info.delete("1.0", tk.END)
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state=tk.DISABLED)
        self.info.yview_moveto(current_y)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "播放同步采集的 HDF5 Episode；省略路径时自动打开 datasets/ 下最新文件。"
        )
    )
    parser.add_argument("path", nargs="?", help="HDF5 文件或数据集目录")
    parser.add_argument("--camera", help="相机名称，默认使用文件中的第一个相机")
    parser.add_argument("--speed", type=float, default=1.0, help="初始播放倍速")
    parser.add_argument("--paused", action="store_true", help="打开后保持暂停")
    parser.add_argument("--depth-min", type=float, default=0.2, help="深度着色下限，米")
    parser.add_argument("--depth-max", type=float, default=2.0, help="深度着色上限，米")
    parser.add_argument(
        "--display-width", type=int, default=560, help="RGB/深度显示宽度"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.speed <= 0.0:
        raise SystemExit("--speed 必须大于 0")
    if args.depth_min < 0.0 or args.depth_max <= args.depth_min:
        raise SystemExit("深度范围必须满足 0 <= depth-min < depth-max")
    if args.display_width < 160:
        raise SystemExit("--display-width 不能小于 160")

    try:
        path = _resolve_episode(args.path)
        episode = Episode(path, args.camera)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"无法打开 Episode: {error}") from error

    print(f"正在播放: {path}")
    try:
        EpisodePlayer(
            episode,
            speed=args.speed,
            paused=args.paused,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            display_width=args.display_width,
        ).run()
    except tk.TclError as error:
        episode.close()
        raise SystemExit(f"无法创建图形窗口，请确认当前终端具有桌面 DISPLAY: {error}") from error


if __name__ == "__main__":
    main()
