import argparse
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, Optional

import cv2
import numpy as np
import requests

from r1lite_env.client import R1LiteClient, decode_image_base64


def _format_scalar(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_vector(value) -> str:
    if value is None:
        return "-"
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v): .3f}" for v in arr) + "]"


def _make_ppm_image(rgb: np.ndarray) -> bytes:
    height, width, _ = rgb.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb.tobytes()


class R1LiteMonitorGUI:
    def __init__(self, root: tk.Tk, server_url: str, image_hz: float, state_period: float):
        self.root = root
        self.client = R1LiteClient(server_url, timeout=2.0)
        self.image_period = max(1.0 / max(image_hz, 1e-3), 0.05)
        self.state_period = max(state_period, 0.1)
        self.stop_event = threading.Event()
        self.latest_packet: Optional[Dict] = None
        self.latest_health: Optional[Dict] = None
        self.packet_lock = threading.Lock()
        self.last_text_update = 0.0
        self.photo_cache: Dict[str, tk.PhotoImage] = {}
        self.last_owner = None
        self.last_teleop_source = None
        self.last_brake_enabled = None
        self.last_faults = []
        self.warned_keys = set()
        self.info_messages = []
        self.warning_messages = []
        self.fault_messages = []

        self.root.title("R1Lite Dual-Arm Monitor")
        self.root.geometry("1920x1120")
        self.root.configure(bg="#e8ecf1")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.summary_vars = {
            "server": tk.StringVar(value=server_url),
            "mode": tk.StringVar(value="-"),
            "owner": tk.StringVar(value="-"),
            "teleop_source": tk.StringVar(value="-"),
            "last_command_age": tk.StringVar(value="-"),
            "brake": tk.StringVar(value="-"),
            "updated": tk.StringVar(value="-"),
        }
        self.left_command_var = tk.StringVar(value="waiting for data")
        self.right_command_var = tk.StringVar(value="waiting for data")
        self.left_state_var = tk.StringVar(value="waiting for data")
        self.right_state_var = tk.StringVar(value="waiting for data")
        self.faults_var = tk.StringVar(value="no faults")

        self.image_labels: Dict[str, ttk.Label] = {}

        self._build_layout()
        self._start_workers()
        self.root.after(100, self._drain_updates)

    def _build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Panel.TLabelframe.Label", font=("TkDefaultFont", 13, "bold"))
        style.configure("Panel.TLabelframe", background="#f8fafc")
        style.configure("TButton", font=("TkDefaultFont", 12))
        style.configure("TLabel", font=("TkDefaultFont", 12))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=0)

        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for idx in range(7):
            top.columnconfigure(idx, weight=1)

        summary_items = [
            ("Server", "server"),
            ("Mode", "mode"),
            ("Owner", "owner"),
            ("Teleop", "teleop_source"),
            ("Cmd Age", "last_command_age"),
            ("Brake", "brake"),
            ("Updated", "updated"),
        ]
        for col, (label, key) in enumerate(summary_items):
            box = ttk.LabelFrame(top, text=label, style="Panel.TLabelframe", padding=8)
            box.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            ttk.Label(box, textvariable=self.summary_vars[key], anchor="center", font=("TkDefaultFont", 12, "bold")).pack(fill="both", expand=True)

        main = ttk.Frame(outer)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=2)
        main.columnconfigure(2, weight=4)
        main.rowconfigure(0, weight=1)

        self._build_arm_panel(main, 0, "Left Arm", self.left_command_var, self.left_state_var, "left_wrist")
        self._build_center_panel(main, 1)
        self._build_arm_panel(main, 2, "Right Arm", self.right_command_var, self.right_state_var, "right_wrist")

        logs_frame = ttk.LabelFrame(outer, text="Logs", style="Panel.TLabelframe", padding=8)
        logs_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        for idx in range(3):
            logs_frame.columnconfigure(idx, weight=1)
        logs_frame.rowconfigure(0, weight=1)

        self.info_text = scrolledtext.ScrolledText(
            logs_frame,
            height=7,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 12),
        )
        self.info_text.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.warning_text = scrolledtext.ScrolledText(
            logs_frame,
            height=7,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 12),
        )
        self.warning_text.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        self.faults_text = scrolledtext.ScrolledText(
            logs_frame,
            height=7,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 12),
        )
        self.faults_text.grid(row=0, column=2, sticky="nsew")
        self._set_text_widget(self.info_text, "info: waiting for telemetry")
        self._set_text_widget(self.warning_text, "warning: none")
        self._set_text_widget(self.faults_text, "fault: none")

    def _build_arm_panel(self, parent, column: int, title: str, command_var: tk.StringVar, state_var: tk.StringVar, image_key: str):
        frame = ttk.LabelFrame(parent, text=title, style="Panel.TLabelframe", padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Command", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=command_var, justify="left", font=("TkFixedFont", 12)).grid(row=1, column=0, sticky="ew", pady=(2, 12))

        ttk.Label(frame, text="State", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        state_message = tk.Message(frame, textvariable=state_var, width=820, anchor="nw", justify="left", font=("TkFixedFont", 12))
        state_message.grid(row=3, column=0, sticky="nsew", pady=(2, 12))

        img_frame = ttk.LabelFrame(frame, text=image_key, style="Panel.TLabelframe", padding=4)
        img_frame.grid(row=4, column=0, sticky="nsew")
        label = ttk.Label(img_frame, text=f"{image_key}\nwaiting for image", anchor="center")
        label.pack(fill="both", expand=True)
        self.image_labels[image_key] = label

    def _build_center_panel(self, parent, column: int):
        frame = ttk.LabelFrame(parent, text="Head Camera And Controls", style="Panel.TLabelframe", padding=8)
        frame.grid(row=0, column=column, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        head_frame = ttk.LabelFrame(frame, text="head", style="Panel.TLabelframe", padding=4)
        head_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        head_label = ttk.Label(head_frame, text="head\nwaiting for image", anchor="center")
        head_label.pack(fill="both", expand=True)
        self.image_labels["head"] = head_label
        self.image_labels["head_left"] = head_label

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Brake Toggle", command=self._toggle_brake).grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(buttons, text="Reset", command=self._reset_robot).grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(buttons, text="Clear Fault", command=self._clear_fault).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(buttons, text="Refresh Now", command=self._refresh_once_async).grid(row=3, column=0, sticky="ew")

        info = ttk.LabelFrame(frame, text="Hints", style="Panel.TLabelframe", padding=8)
        info.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        hint = (
            "Images update at high rate.\n"
            "State and command text update every 0.5s.\n"
            "Maintenance actions use owner=debug."
        )
        ttk.Label(info, text=hint, justify="left", font=("TkDefaultFont", 12)).pack(fill="both", expand=True)

    def _start_workers(self):
        self.image_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.image_thread.start()

    def _poll_loop(self):
        next_health = 0.0
        while not self.stop_event.is_set():
            start = time.time()
            try:
                packet = self.client.get_state()
                health = None
                if start >= next_health:
                    health = self.client.get_health()
                    next_health = start + self.state_period
                with self.packet_lock:
                    self.latest_packet = packet
                    if health is not None:
                        self.latest_health = health
            except requests.RequestException as exc:
                self._set_fault_text(f"request error: {exc}")
            elapsed = time.time() - start
            time.sleep(max(0.0, self.image_period - elapsed))

    def _set_fault_text(self, text: str):
        self.root.after(0, lambda: self._append_log("fault", text))

    def _set_text_widget(self, widget: scrolledtext.ScrolledText, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def _refresh_log_views(self):
        self._set_text_widget(self.info_text, "\n".join(self.info_messages[-80:]) if self.info_messages else "info: none")
        self._set_text_widget(
            self.warning_text,
            "\n".join(self.warning_messages[-80:]) if self.warning_messages else "warning: none",
        )
        self._set_text_widget(self.faults_text, "\n".join(self.fault_messages[-80:]) if self.fault_messages else "fault: none")

    def _append_log(self, level: str, message: str):
        timestamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        if level == "info":
            if timestamped not in self.info_messages:
                self.info_messages.append(timestamped)
        elif level == "warning":
            if timestamped not in self.warning_messages:
                self.warning_messages.append(timestamped)
        else:
            if timestamped not in self.fault_messages:
                self.fault_messages.append(timestamped)
        self._refresh_log_views()

    def _sync_logs_from_health(self, packet: Dict, health: Dict):
        owner = packet.get("meta", {}).get("command_owner")
        teleop_source = packet.get("meta", {}).get("active_teleop_source")
        brake_enabled = packet.get("meta", {}).get("brake_enabled")

        if owner != self.last_owner:
            self._append_log("info", f"command owner changed: {self.last_owner} -> {owner}")
            self.last_owner = owner
        if teleop_source != self.last_teleop_source:
            self._append_log("info", f"teleop source changed: {self.last_teleop_source} -> {teleop_source}")
            if owner == "teleop" and teleop_source == "spacemouse":
                self._append_log("info", "SpaceMouse teleop active; startup calibration completed on teleop client")
            self.last_teleop_source = teleop_source
        if brake_enabled != self.last_brake_enabled:
            state = "enabled" if brake_enabled else "disabled"
            self._append_log("info", f"brake {state}")
            self.last_brake_enabled = brake_enabled

        freshness = health.get("freshness", {})
        validity = packet.get("meta", {}).get("validity", {})
        warning_keys = []
        for key, value in freshness.items():
            if value is False:
                warning_keys.append(f"freshness:{key}")
        for side in ("left", "right"):
            for key, value in validity.get(side, {}).items():
                if value is False:
                    warning_keys.append(f"validity:{side}:{key}")
        for key, value in validity.get("images", {}).items():
            if value is False:
                warning_keys.append(f"validity:image:{key}")

        current_warning_keys = set(warning_keys)
        new_warning_keys = current_warning_keys - self.warned_keys
        cleared_warning_keys = self.warned_keys - current_warning_keys
        for key in sorted(new_warning_keys):
            self._append_log("warning", key)
        for key in sorted(cleared_warning_keys):
            self._append_log("info", f"warning cleared: {key}")
        self.warned_keys = current_warning_keys

        faults = sorted(str(item) for item in health.get("faults", []))
        if faults != self.last_faults:
            if not faults:
                self._append_log("info", "all faults cleared")
            else:
                for fault in faults:
                    if fault not in self.last_faults:
                        self._append_log("fault", fault)
            self.last_faults = faults

    def _drain_updates(self):
        packet = None
        health = None
        with self.packet_lock:
            if self.latest_packet is not None:
                packet = self.latest_packet
                self.latest_packet = None
            if self.latest_health is not None:
                health = self.latest_health
                self.latest_health = None

        if packet is not None:
            self._update_images(packet.get("images", {}))
            now = time.time()
            if now - self.last_text_update >= self.state_period:
                self.last_text_update = now
                self._update_text(packet, health)

        self.root.after(100, self._drain_updates)

    def _update_images(self, images: Dict[str, Optional[str]]):
        for key, label in self.image_labels.items():
            rgb = decode_image_base64(images.get(key))
            if rgb is None:
                rgb = np.zeros((288, 384, 3), dtype=np.uint8)
            else:
                rgb = cv2.resize(rgb, (384, 288), interpolation=cv2.INTER_AREA)
            ppm = _make_ppm_image(rgb)
            photo = tk.PhotoImage(data=ppm, format="PPM")
            self.photo_cache[key] = photo
            label.configure(image=photo, text="")

    def _update_text(self, packet: Dict, health: Optional[Dict]):
        meta = packet.get("meta", {})
        command_owner = meta.get("command_owner")
        teleop_source = meta.get("active_teleop_source")
        brake_enabled = meta.get("brake_enabled")
        current_health = health or meta.get("health", {})

        self.summary_vars["mode"].set(str(meta.get("mode", "-")))
        self.summary_vars["owner"].set(str(command_owner))
        self.summary_vars["teleop_source"].set(str(teleop_source))
        self.summary_vars["brake"].set("enabled" if brake_enabled else "disabled")
        cmd_age = current_health.get("last_command_age_sec")
        self.summary_vars["last_command_age"].set(_format_scalar(cmd_age))
        self.summary_vars["updated"].set(time.strftime("%H:%M:%S"))

        commands = meta.get("commands", {})
        state = packet.get("state", {})
        self.left_command_var.set(self._format_command_text("left", commands.get("left", {}), command_owner, teleop_source))
        self.right_command_var.set(self._format_command_text("right", commands.get("right", {}), command_owner, teleop_source))
        self.left_state_var.set(self._format_arm_state_text("left", state.get("left", {}), packet))
        self.right_state_var.set(self._format_arm_state_text("right", state.get("right", {}), packet))

        self._sync_logs_from_health(packet, current_health)

    def _format_command_text(self, side: str, command: Dict, owner, teleop_source) -> str:
        lines = [
            f"owner: {_format_scalar(owner)}",
            f"teleop_source: {_format_scalar(teleop_source)}",
            f"preset: {_format_scalar(command.get('preset'))}",
            f"gripper_target: {_format_scalar(command.get('gripper'))}",
            f"tcp_target: {_format_vector(command.get('desired_pose'))}",
            f"joint_target: {_format_vector(command.get('desired_joint'))}",
            f"updated_at: {_format_scalar(command.get('updated_at'))}",
            f"last_sent: {_format_scalar(command.get('last_sent_target'))}",
        ]
        return "\n".join(lines)

    def _format_arm_state_text(self, side: str, arm_state: Dict, packet: Dict) -> str:
        validity = packet.get("meta", {}).get("validity", {}).get(side, {})
        lines = [
            f"joint_pos: {_format_vector(arm_state.get('joint_pos'))}",
            f"joint_vel: {_format_vector(arm_state.get('joint_vel'))}",
            f"joint_effort: {_format_vector(arm_state.get('joint_effort'))}",
            f"gripper_pose: {_format_vector(arm_state.get('gripper_pose'))}",
            f"tcp_pose: {_format_vector(arm_state.get('tcp_pose'))}",
            f"tcp_vel: {_format_vector(arm_state.get('tcp_vel'))}",
            f"tcp_force: {_format_vector(arm_state.get('tcp_force'))}",
            f"tcp_torque: {_format_vector(arm_state.get('tcp_torque'))}",
            f"state_preset: {_format_scalar(arm_state.get('preset'))}",
            f"validity: {validity}",
        ]
        return "\n".join(lines)

    def _run_action_async(self, fn, success_text: str):
        def worker():
            try:
                fn()
                self._refresh_once()
            except requests.RequestException as exc:
                self.root.after(0, lambda: messagebox.showerror("Request failed", str(exc)))
                return
            self.root.after(0, lambda: messagebox.showinfo("Action sent", success_text))

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_brake(self):
        enabled = self.summary_vars["brake"].get() != "enabled"
        self._run_action_async(lambda: self.client.brake(enabled), f"brake set to {enabled}")

    def _reset_robot(self):
        self._run_action_async(lambda: self.client.reset(owner="debug"), "reset sent")

    def _clear_fault(self):
        self._run_action_async(lambda: self.client.clear_fault(owner="debug"), "clear_fault sent")

    def _refresh_once(self):
        try:
            packet = self.client.get_state()
            health = self.client.get_health()
        except requests.RequestException:
            return
        self.root.after(0, lambda: self._update_images(packet.get("images", {})))
        self.root.after(0, lambda: self._update_text(packet, health))

    def _refresh_once_async(self):
        threading.Thread(target=self._refresh_once, daemon=True).start()

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="R1Lite dual-arm monitor GUI")
    parser.add_argument("--server-url", default="http://127.0.0.1:8001/")
    parser.add_argument("--image-hz", type=float, default=5.0, help="Polling rate for state packets used to refresh images")
    parser.add_argument("--state-period", type=float, default=0.5, help="Refresh period for text state panels")
    args = parser.parse_args()

    root = tk.Tk()
    R1LiteMonitorGUI(root=root, server_url=args.server_url, image_hz=args.image_hz, state_period=args.state_period)
    root.mainloop()


if __name__ == "__main__":
    main()
