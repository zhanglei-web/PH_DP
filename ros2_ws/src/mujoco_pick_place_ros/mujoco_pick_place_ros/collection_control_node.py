"""Small interactive console for controlling synchronized episode recording."""

from __future__ import annotations

import sys
import termios
from threading import Thread
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


HELP = """采集快捷键（无需按 Enter）：
  s  开始采集当前轨迹
  f  保存当前轨迹并 reset
  d  丢弃当前轨迹并 reset
  q  完成任务并退出采集控制台
  t  输入新的任务名称
  h  显示本帮助
"""

SHORTCUT_COMMANDS = {
    "s": "start",
    "f": "save",
    "d": "discard",
    "q": "finish",
}


class CollectionControlNode(Node):
    def __init__(self) -> None:
        super().__init__("collection_control")
        self._publisher = self.create_publisher(
            String, "/mujoco/collection/command", 10
        )
        self._status_subscription = self.create_subscription(
            String, "/mujoco/collection/status", self._on_status, 10
        )
        self.recording = False

    def send(self, command: str) -> None:
        verb = command.partition(" ")[0].lower()
        if verb == "start":
            self.recording = True
        elif verb in {"save", "discard"}:
            self.recording = False
        self._publisher.publish(String(data=command))

    def _on_status(self, message: String) -> None:
        if message.data.startswith("recording started:"):
            self.recording = True
        elif message.data.startswith(("saving ", "discarding ")):
            self.recording = False
        print(f"\n[collector] {message.data}", flush=True)


def _run_shortcut_console(node: CollectionControlNode) -> None:
    descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(descriptor)
    print(HELP, flush=True)
    try:
        tty.setcbreak(descriptor)
        while rclpy.ok():
            key = sys.stdin.read(1).lower()
            if key in {"\n", "\r", " "}:
                continue
            if key == "h":
                print(f"\n{HELP}", flush=True)
                continue
            if key == "t":
                termios.tcsetattr(descriptor, termios.TCSADRAIN, original_settings)
                try:
                    task_name = input("\n请输入任务名称: ").strip()
                finally:
                    tty.setcbreak(descriptor)
                if task_name:
                    node.send(f"task {task_name}")
                continue
            command = SHORTCUT_COMMANDS.get(key)
            if command is None:
                print(f"\n未知快捷键 '{key}'，按 h 查看帮助。", flush=True)
                continue
            if key == "q" and node.recording:
                print(
                    "\n当前轨迹仍在采集，请先按 f 保存或按 d 丢弃。",
                    flush=True,
                )
                continue
            node.send(command)
            if key == "q":
                time.sleep(0.5)
                return
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original_settings)


def _run_line_console(node: CollectionControlNode) -> None:
    """Fallback for redirected stdin and terminals without cbreak support."""
    print(HELP)
    while rclpy.ok():
        command = input("collection> ").strip()
        if not command:
            continue
        if command in SHORTCUT_COMMANDS:
            command = SHORTCUT_COMMANDS[command]
        if command == "h" or command == "help":
            print(HELP)
            continue
        verb = command.partition(" ")[0].lower()
        if verb not in {"task", "start", "save", "discard", "finish"}:
            print("未知命令，输入 help 查看帮助。")
            continue
        if verb == "finish" and node.recording:
            print("当前轨迹仍在采集，请先保存或丢弃。")
            continue
        node.send(command)
        if verb == "finish":
            time.sleep(0.5)
            return


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CollectionControlNode()
    spin_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        if sys.stdin.isatty():
            _run_shortcut_console(node)
        else:
            _run_line_console(node)
    except (EOFError, KeyboardInterrupt):
        print("\n采集控制台已关闭；正在采集的轨迹不会自动保存。")
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
