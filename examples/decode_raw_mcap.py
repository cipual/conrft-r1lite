#!/usr/bin/env python3
"""Inspect and decode R1Lite RAW rosbag2 MCAP episodes.

Examples:
    python examples/decode_raw_mcap.py list data/RAW/r1lite_dual_mango_box
    python examples/decode_raw_mcap.py sample data/RAW/r1lite_dual_mango_box/RB..._RAW --topic /hdas/feedback_arm_right
    python examples/decode_raw_mcap.py export-images data/RAW/r1lite_dual_mango_box/RB..._RAW --topic /hdas/camera_wrist_right/color/image_raw/compressed
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


def is_raw_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.yaml").exists() and bool(list(path.glob("*.mcap")))


def resolve_raw_episode_dirs(paths: Sequence[str], recursive: bool = False) -> List[Path]:
    episodes: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if is_raw_episode_dir(path):
            episodes.append(path)
            continue
        if path.is_file() and path.suffix == ".mcap":
            if (path.parent / "metadata.yaml").exists():
                episodes.append(path.parent.resolve())
                continue
            raise ValueError(f"MCAP file has no sibling metadata.yaml: {path}")
        if not path.is_dir():
            raise ValueError(f"Input is not a RAW episode, MCAP file, or parent directory: {path}")

        candidates = path.rglob("*_RAW") if recursive else path.glob("*_RAW")
        found = sorted(candidate.resolve() for candidate in candidates if is_raw_episode_dir(candidate))
        if not found:
            mode = " recursively" if recursive else ""
            raise FileNotFoundError(f"No RAW episode directories found{mode} under {path}")
        episodes.extend(found)

    unique: List[Path] = []
    seen = set()
    for episode in episodes:
        if episode not in seen:
            unique.append(episode)
            seen.add(episode)
    return unique


def load_metadata(episode_dir: Path) -> dict:
    with (episode_dir / "metadata.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bag_info(metadata: dict) -> dict:
    return metadata.get("rosbag2_bagfile_information", {})


def topic_rows(metadata: dict) -> List[dict]:
    rows = []
    for item in bag_info(metadata).get("topics_with_message_count", []):
        topic_meta = item.get("topic_metadata", {})
        rows.append(
            {
                "name": topic_meta.get("name", ""),
                "type": topic_meta.get("type", ""),
                "count": int(item.get("message_count", 0)),
                "serialization_format": topic_meta.get("serialization_format", ""),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def print_topic_table(episode_dir: Path, metadata: dict, min_count: int = 0) -> None:
    info = bag_info(metadata)
    duration_ns = int(info.get("duration", {}).get("nanoseconds", 0))
    duration_s = duration_ns / 1e9 if duration_ns else 0.0
    print(f"\n{episode_dir}")
    print(f"  messages: {int(info.get('message_count', 0))}  duration: {duration_s:.3f}s")
    print("  count      hz      type                                      topic")
    print("  ---------  ------  ----------------------------------------  -----")
    for row in topic_rows(metadata):
        if row["count"] < min_count:
            continue
        hz = (row["count"] / duration_s) if duration_s > 0 else 0.0
        print(f"  {row['count']:9d}  {hz:6.2f}  {row['type'][:40]:40s}  {row['name']}")


def prepare_rosbags_input_dir(episode_dir: Path, metadata: dict) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Build a compatibility directory when metadata points to a renamed MCAP file."""
    info = bag_info(metadata)
    relative_paths = [str(x) for x in info.get("relative_file_paths", [])]
    if not relative_paths:
        return episode_dir, None

    missing = [name for name in relative_paths if not (episode_dir / name).exists()]
    if not missing:
        return episode_dir, None

    actual_mcap_files = sorted(episode_dir.glob("*.mcap"))
    if len(actual_mcap_files) != 1:
        raise FileNotFoundError(
            f"metadata.yaml expects missing MCAP files {missing} under {episode_dir}; "
            "automatic compatibility mode only supports exactly one actual .mcap file."
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="r1lite_raw_decode_", dir="/tmp")
    compat_dir = Path(temp_dir.name)
    shutil.copy2(episode_dir / "metadata.yaml", compat_dir / "metadata.yaml")
    actual_mcap = actual_mcap_files[0]
    for relative_name in relative_paths:
        target = compat_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(actual_mcap)
    return compat_dir, temp_dir


def load_rosbags_reader():
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("rosbags is required for decoding. Install it with: python -m pip install rosbags") from exc
    return AnyReader


def to_builtin(value: Any, max_sequence: int = 24) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "preview_hex": value[:32].hex()}
    if isinstance(value, bytearray):
        return {"bytes": len(value), "preview_hex": bytes(value[:32]).hex()}
    if isinstance(value, memoryview):
        raw = value.tobytes()
        return {"bytes": len(raw), "preview_hex": raw[:32].hex()}
    if isinstance(value, (list, tuple)):
        values = list(value)
        out = [to_builtin(item, max_sequence=max_sequence) for item in values[:max_sequence]]
        if len(values) > max_sequence:
            out.append(f"... {len(values) - max_sequence} more")
        return out
    if hasattr(value, "tolist"):
        values = value.tolist()
        if isinstance(values, list) and len(values) > max_sequence:
            return values[:max_sequence] + [f"... {len(values) - max_sequence} more"]
        return values
    if is_dataclass(value):
        return {name: to_builtin(getattr(value, name), max_sequence=max_sequence) for name in value.__dataclass_fields__}
    slots = getattr(value, "__slots__", None)
    if slots:
        return {name: to_builtin(getattr(value, name), max_sequence=max_sequence) for name in slots if hasattr(value, name)}
    if hasattr(value, "__dict__"):
        return {name: to_builtin(item, max_sequence=max_sequence) for name, item in vars(value).items()}
    return repr(value)


def summarize_ros_message(topic: str, msg: Any) -> dict:
    summary = {"topic": topic, "decoded": True}
    if hasattr(msg, "pose"):
        pose = msg.pose
        summary["pose"] = {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation_xyzw": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        }
        if hasattr(msg, "header"):
            summary["header"] = to_builtin(msg.header)
        return summary
    if hasattr(msg, "position") and hasattr(msg, "velocity") and hasattr(msg, "effort"):
        summary["joint_state"] = {
            "name": to_builtin(getattr(msg, "name", [])),
            "position": to_builtin(getattr(msg, "position", [])),
            "velocity": to_builtin(getattr(msg, "velocity", [])),
            "effort": to_builtin(getattr(msg, "effort", [])),
        }
        if hasattr(msg, "header"):
            summary["header"] = to_builtin(msg.header)
        return summary
    if hasattr(msg, "format") and hasattr(msg, "data"):
        summary["compressed_image"] = {
            "format": str(msg.format),
            "bytes": len(bytes(msg.data)),
        }
        if hasattr(msg, "header"):
            summary["header"] = to_builtin(msg.header)
        return summary
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "encoding") and hasattr(msg, "data"):
        summary["image"] = {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": str(msg.encoding),
            "step": int(getattr(msg, "step", 0)),
            "bytes": len(bytes(msg.data)),
        }
        if hasattr(msg, "header"):
            summary["header"] = to_builtin(msg.header)
        return summary
    if hasattr(msg, "k") and hasattr(msg, "d"):
        summary["camera_info"] = {
            "height": int(getattr(msg, "height", 0)),
            "width": int(getattr(msg, "width", 0)),
            "k": to_builtin(getattr(msg, "k")),
            "d": to_builtin(getattr(msg, "d")),
        }
        if hasattr(msg, "header"):
            summary["header"] = to_builtin(msg.header)
        return summary
    summary["message"] = to_builtin(msg)
    return summary


def iter_decoded_messages(episode_dir: Path, topics: Sequence[str]) -> Iterable[Tuple[str, int, Any]]:
    metadata = load_metadata(episode_dir)
    reader_dir, temp_dir = prepare_rosbags_input_dir(episode_dir, metadata)
    AnyReader = load_rosbags_reader()
    try:
        with AnyReader([reader_dir]) as reader:
            topic_set = set(topics)
            connections = [conn for conn in reader.connections if not topic_set or conn.topic in topic_set]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                yield connection.topic, int(timestamp), reader.deserialize(rawdata, connection.msgtype)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def cmd_list(args: argparse.Namespace) -> None:
    for episode_dir in resolve_raw_episode_dirs(args.inputs, recursive=args.recursive):
        print_topic_table(episode_dir, load_metadata(episode_dir), min_count=args.min_count)


def cmd_sample(args: argparse.Namespace) -> None:
    episodes = resolve_raw_episode_dirs([args.input], recursive=False)
    if len(episodes) != 1:
        raise ValueError("sample expects one RAW episode directory or one MCAP file")
    seen = 0
    for topic, timestamp, msg in iter_decoded_messages(episodes[0], args.topic):
        record = {
            "timestamp_ns": timestamp,
            **summarize_ros_message(topic, msg),
        }
        print(json.dumps(record, ensure_ascii=False, indent=2))
        seen += 1
        if seen >= args.limit:
            break
    if seen == 0:
        topics = ", ".join(args.topic) if args.topic else "(all topics)"
        print(f"No messages decoded for {topics}")


def decode_image_to_rgb(msg: Any):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for image export: python -m pip install opencv-python numpy") from exc

    if hasattr(msg, "format") and hasattr(msg, "data"):
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode sensor_msgs/CompressedImage")
        return bgr

    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "encoding") and hasattr(msg, "data"):
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if encoding == "bgr8":
            return raw.reshape(height, width, 3)
        if encoding == "rgb8":
            return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(raw.reshape(height, width), cv2.COLOR_GRAY2BGR)
        if encoding in ("16uc1", "mono16"):
            depth = raw.view(np.uint16).reshape(height, width)
            depth8 = cv2.convertScaleAbs(depth, alpha=255.0 / max(float(depth.max()), 1.0))
            return cv2.cvtColor(depth8, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Unsupported sensor_msgs/Image encoding: {msg.encoding}")

    raise ValueError("Selected message is not a sensor_msgs/Image or sensor_msgs/CompressedImage")


def cmd_export_images(args: argparse.Namespace) -> None:
    episodes = resolve_raw_episode_dirs([args.input], recursive=False)
    if len(episodes) != 1:
        raise ValueError("export-images expects one RAW episode directory or one MCAP file")
    if not args.topic:
        raise ValueError("export-images requires --topic")
    import cv2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    stride_count = 0
    for topic, timestamp, msg in iter_decoded_messages(episodes[0], [args.topic]):
        if stride_count % args.stride == 0:
            image = decode_image_to_rgb(msg)
            topic_name = topic.strip("/").replace("/", "__")
            output_path = output_dir / f"{topic_name}_{timestamp}.jpg"
            ok = cv2.imwrite(str(output_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write image: {output_path}")
            written += 1
            if written >= args.limit:
                break
        stride_count += 1
    print(f"Wrote {written} images to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and decode R1Lite RAW rosbag2 MCAP data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List RAW episodes and their topics from metadata.yaml.")
    list_parser.add_argument("inputs", nargs="+", help="RAW episode dirs, MCAP files, or parent directories.")
    list_parser.add_argument("--recursive", action="store_true", help="Search parent directories recursively for *_RAW episodes.")
    list_parser.add_argument("--min-count", type=int, default=0, help="Only show topics with at least this many messages.")
    list_parser.set_defaults(func=cmd_list)

    sample_parser = subparsers.add_parser("sample", help="Decode and print a few ROS messages as JSON.")
    sample_parser.add_argument("input", help="One RAW episode dir or MCAP file.")
    sample_parser.add_argument("--topic", action="append", default=[], help="Topic to decode. Can be passed multiple times.")
    sample_parser.add_argument("--limit", type=int, default=5, help="Maximum decoded messages to print.")
    sample_parser.set_defaults(func=cmd_sample)

    image_parser = subparsers.add_parser("export-images", help="Export image messages from one topic as JPG files.")
    image_parser.add_argument("input", help="One RAW episode dir or MCAP file.")
    image_parser.add_argument("--topic", required=True, help="Image topic to export.")
    image_parser.add_argument("--output-dir", default="raw_mcap_images", help="Directory for exported JPG files.")
    image_parser.add_argument("--limit", type=int, default=20, help="Maximum images to write.")
    image_parser.add_argument("--stride", type=int, default=1, help="Write every Nth image message.")
    image_parser.set_defaults(func=cmd_export_images)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "stride") and args.stride <= 0:
        parser.error("--stride must be > 0")
    if hasattr(args, "limit") and args.limit <= 0:
        parser.error("--limit must be > 0")
    args.func(args)


if __name__ == "__main__":
    main()
