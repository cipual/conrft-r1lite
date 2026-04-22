#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse


DEFAULT_TASK_DESC = "左臂抓住白色的框放在右臂的周围，右臂抓住发红的芒果，把它放入框内，然后左右机械臂复位。"

DEFAULT_SPARSE_SUBTASKS = [
    "left arm grasps the white box",
    "left arm positions the white box around the right arm",
    "right arm grasps the red mango",
    "right arm places the red mango into the white box",
    "both robot arms return to the reset pose",
]

DEFAULT_DENSE_SUBTASKS = [
    "left arm approaches the white box",
    "left gripper closes on the white box",
    "left arm moves the white box",
    "the white box is positioned around the right arm",
    "right arm approaches the red mango",
    "right gripper closes on the red mango",
    "right arm moves the mango above the white box",
    "the mango is released inside the white box",
    "both arms move away from the objects",
    "both arms reach the reset pose",
]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>R1Lite SARM LeRobot Annotation</title>
  <style>
    :root {
      --ink: #1c1917;
      --muted: #78716c;
      --line: #d6d3d1;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
    }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% 8%, rgba(20,184,166,0.20), transparent 28%),
        linear-gradient(135deg, #fff7ed 0%, #f0fdfa 100%);
      color: var(--ink);
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    }
    header { padding: 22px 28px 10px; }
    h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.03em; }
    .task { color: var(--muted); max-width: 1120px; line-height: 1.45; }
    main {
      display: grid;
      grid-template-columns: minmax(420px, 54vw) 1fr;
      gap: 18px;
      padding: 12px 28px 28px;
    }
    .card {
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(120,113,108,0.25);
      border-radius: 18px;
      box-shadow: 0 18px 55px rgba(28,25,23,0.10);
      overflow: hidden;
    }
    .video-card { padding: 16px; }
    video { width: 100%; max-height: 70vh; background: #111; border-radius: 14px; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
    select, input, button {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
    }
    button { cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); font-weight: 700; }
    button.secondary { color: var(--accent); background: var(--accent-soft); border-color: #99f6e4; }
    button.ghost { color: var(--muted); background: #fafaf9; border-color: var(--line); }
    .meta { font-size: 14px; color: var(--muted); }
    .panel { padding: 14px 16px; max-height: 82vh; overflow: auto; }
    h2 { margin: 10px 0 8px; font-size: 20px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 18px; }
    th, td {
      text-align: left;
      border-bottom: 1px solid #e7e5e4;
      padding: 7px 4px;
      vertical-align: top;
      font-size: 14px;
    }
    th { color: var(--muted); font-weight: 700; }
    td.task-name { width: 34%; line-height: 1.35; }
    input.frame { width: 76px; }
    input.notes { width: 100%; box-sizing: border-box; }
    .status { min-height: 22px; color: var(--accent); font-weight: 700; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      .panel { max-height: unset; }
    }
  </style>
</head>
<body>
  <header>
    <h1>R1Lite SARM LeRobot Annotation</h1>
    <div class="task" id="task"></div>
  </header>
  <main>
    <section class="card video-card">
      <div class="row">
        <label>Episode</label>
        <select id="episode"></select>
        <button class="secondary" id="save">Save to LeRobot</button>
        <button class="ghost" id="reload">Reload</button>
      </div>
      <video id="video" controls preload="metadata"></video>
      <div class="row">
        <button class="ghost" id="jumpStart">Jump to episode start</button>
        <button class="ghost" id="jumpEnd">Jump to episode end</button>
      </div>
      <div class="row meta">
        <span id="episodeMeta"></span>
        <span id="currentMeta"></span>
      </div>
      <div class="status" id="status"></div>
    </section>
    <section class="card panel">
      <h2>Sparse Subtask Segments</h2>
      <table id="sparse"></table>
      <h2>Dense Subtask Segments</h2>
      <table id="dense"></table>
    </section>
  </main>
  <script>
    let data = null;
    let currentEpisodeIndex = 0;
    const el = (id) => document.getElementById(id);
    const video = el("video");

    async function loadData() {
      const res = await fetch("/api/annotations");
      data = await res.json();
      el("task").textContent = `${data.task_desc || ""} | dataset=${data.dataset_root}`;
      renderEpisodeSelect();
      renderEpisode();
    }

    function currentEpisode() {
      return data.episodes[currentEpisodeIndex];
    }

    function localTime() {
      const ep = currentEpisode();
      return Math.max(0, Math.min(ep.duration_sec, video.currentTime - ep.video_start));
    }

    function currentFrame() {
      const ep = currentEpisode();
      return Math.max(0, Math.min(ep.num_frames - 1, Math.round(localTime() * data.fps)));
    }

    function frameToGlobalTime(frame) {
      return currentEpisode().video_start + frame / data.fps;
    }

    function annotationList(kind) {
      const ep = currentEpisode();
      ep.annotations = ep.annotations || {};
      ep.annotations[kind] = ep.annotations[kind] || [];
      return ep.annotations[kind];
    }

    function getAnnotation(kind, name) {
      const list = annotationList(kind);
      let found = list.find((item) => item.name === name);
      if (!found) {
        found = {name, start_frame: null, end_frame: null, start_time_sec: null, end_time_sec: null, notes: ""};
        list.push(found);
      }
      return found;
    }

    function renderEpisodeSelect() {
      el("episode").innerHTML = "";
      data.episodes.forEach((ep, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `${idx}: episode ${ep.episode_index}`;
        el("episode").appendChild(opt);
      });
      el("episode").value = currentEpisodeIndex;
    }

    function renderEpisode() {
      const ep = currentEpisode();
      let didInitialSeek = false;
      const jumpToStartOnce = () => {
        if (didInitialSeek) return;
        didInitialSeek = true;
        seekGlobal(ep.video_start);
      };
      video.pause();
      video.onloadedmetadata = null;
      video.oncanplay = null;
      video.src = ep.video_url;
      video.load();
      video.onloadedmetadata = jumpToStartOnce;
      video.oncanplay = jumpToStartOnce;
      el("episodeMeta").textContent =
        `episode_index=${ep.episode_index} | frames=${ep.num_frames} | local=${ep.duration_sec.toFixed(2)}s | video=${ep.video_path}`;
      renderTable("sparse", data.sparse_subtasks || []);
      renderTable("dense", data.dense_subtasks || []);
      updateCurrentMeta();
    }

    function renderTable(kind, tasks) {
      const table = el(kind);
      table.innerHTML = "";
      const head = document.createElement("tr");
      head.innerHTML = "<th>Subtask</th><th>Start</th><th>End</th><th>Mark</th><th>Notes</th>";
      table.appendChild(head);
      tasks.forEach((name) => {
        const taskIndex = tasks.indexOf(name);
        const ann = getAnnotation(kind, name);
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="task-name">${name}</td>
          <td><input class="frame start" type="number" min="0" step="1"></td>
          <td><input class="frame end" type="number" min="0" step="1"></td>
          <td>
            <button class="secondary set-start">Set start</button>
            <button class="secondary set-end">Set end</button>
            <button class="ghost seek-start">Seek start</button>
            <button class="ghost seek-end">Seek end</button>
            <button class="ghost clear">Clear</button>
          </td>
          <td><input class="notes" type="text"></td>
        `;
        const startInput = tr.querySelector("input.start");
        const endInput = tr.querySelector("input.end");
        const notesInput = tr.querySelector("input.notes");
        startInput.value = ann.start_frame === null || ann.start_frame === undefined ? "" : ann.start_frame;
        endInput.value = ann.end_frame === null || ann.end_frame === undefined ? "" : ann.end_frame;
        notesInput.value = ann.notes || "";
        function setFrame(field, value) {
          const ep = currentEpisode();
          const frame = value === "" ? null : Math.max(0, Math.min(ep.num_frames - 1, Number(value)));
          ann[field] = frame;
          const timeField = field === "start_frame" ? "start_time_sec" : "end_time_sec";
          ann[timeField] = frame === null ? null : frame / data.fps;
        }
        function setAnnotationFrame(targetAnn, field, frame) {
          const ep = currentEpisode();
          const clamped = frame === null || frame === undefined ? null : Math.max(0, Math.min(ep.num_frames - 1, Number(frame)));
          targetAnn[field] = clamped;
          const timeField = field === "start_frame" ? "start_time_sec" : "end_time_sec";
          targetAnn[timeField] = clamped === null ? null : clamped / data.fps;
        }
        function propagateEndToNextStart(frame) {
          if (frame === null || frame === undefined || taskIndex + 1 >= tasks.length) return;
          const nextName = tasks[taskIndex + 1];
          const nextAnn = getAnnotation(kind, nextName);
          if (nextAnn.start_frame === null || nextAnn.start_frame === undefined || nextAnn.start_frame === "") {
            setAnnotationFrame(nextAnn, "start_frame", frame);
            renderTable(kind, tasks);
          }
        }
        startInput.addEventListener("input", () => setFrame("start_frame", startInput.value));
        endInput.addEventListener("input", () => {
          setFrame("end_frame", endInput.value);
          propagateEndToNextStart(ann.end_frame);
        });
        notesInput.addEventListener("input", () => { ann.notes = notesInput.value; });
        tr.querySelector("button.set-start").addEventListener("click", () => {
          const frame = currentFrame();
          setFrame("start_frame", frame);
          startInput.value = frame;
        });
        tr.querySelector("button.set-end").addEventListener("click", () => {
          const frame = currentFrame();
          setFrame("end_frame", frame);
          endInput.value = frame;
          propagateEndToNextStart(frame);
        });
        tr.querySelector("button.seek-start").addEventListener("click", () => {
          if (ann.start_frame !== null && ann.start_frame !== undefined) video.currentTime = frameToGlobalTime(ann.start_frame);
        });
        tr.querySelector("button.seek-end").addEventListener("click", () => {
          if (ann.end_frame !== null && ann.end_frame !== undefined) video.currentTime = frameToGlobalTime(ann.end_frame);
        });
        tr.querySelector("button.clear").addEventListener("click", () => {
          ann.start_frame = null;
          ann.end_frame = null;
          ann.start_time_sec = null;
          ann.end_time_sec = null;
          ann.notes = "";
          startInput.value = "";
          endInput.value = "";
          notesInput.value = "";
        });
        table.appendChild(tr);
      });
    }

    function updateCurrentMeta() {
      if (!data) return;
      const ep = currentEpisode();
      el("currentMeta").textContent =
        `global_time=${video.currentTime.toFixed(2)}s local_time=${localTime().toFixed(2)}s frame=${currentFrame()}`;
      if (video.currentTime > ep.video_end + 0.05) {
        video.pause();
        video.currentTime = ep.video_end;
      }
    }

    function seekGlobal(timeSec) {
      const target = Number(timeSec || 0);
      try {
        if (typeof video.fastSeek === "function") {
          video.fastSeek(target);
        } else {
          video.currentTime = target;
        }
      } catch (err) {
        video.currentTime = target;
      }
      updateCurrentMeta();
    }

    async function saveData() {
      data.updated_at = new Date().toISOString();
      const res = await fetch("/api/annotations", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data, null, 2)
      });
      const out = await res.json();
      el("status").textContent = out.ok ? `Saved: ${out.path}` : `Save failed: ${out.error}`;
      setTimeout(() => el("status").textContent = "", 4500);
    }

    el("episode").addEventListener("change", () => {
      currentEpisodeIndex = Number(el("episode").value);
      renderEpisode();
    });
    el("save").addEventListener("click", saveData);
    el("reload").addEventListener("click", loadData);
    el("jumpStart").addEventListener("click", () => seekGlobal(currentEpisode().video_start));
    el("jumpEnd").addEventListener("click", () => seekGlobal(currentEpisode().video_end));
    video.addEventListener("timeupdate", updateCurrentMeta);
    video.addEventListener("loadedmetadata", updateCurrentMeta);
    loadData();
  </script>
</body>
</html>
"""


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "manual_annotate_sarm.py writes LeRobot parquet metadata and requires pandas + pyarrow. "
            "Run it in the `lerobot` conda env after installing the dataset extras, e.g. "
            "`pip install 'lerobot[dataset]' pandas pyarrow`, or install the equivalent deps in your local LeRobot env."
        ) from exc
    return pd


def locate_dataset_root(dataset_root: Optional[str], repo_id: Optional[str], root: Optional[str]) -> Path:
    candidates: List[Path] = []
    if dataset_root:
        candidates.append(Path(dataset_root).expanduser())
    if root:
        root_path = Path(root).expanduser()
        candidates.append(root_path)
        if repo_id:
            candidates.append(root_path / repo_id)
    if repo_id:
        hf_home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot")).expanduser()
        candidates.append(hf_home / repo_id)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "meta" / "info.json").exists():
            return candidate
    checked = "\n".join(f"  - {path.expanduser()}" for path in candidates)
    raise FileNotFoundError(f"Could not locate a LeRobotDataset root with meta/info.json. Checked:\n{checked}")


def load_info(dataset_root: Path) -> Dict:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_parquet_paths(dataset_root: Path) -> List[Path]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {dataset_root / 'meta' / 'episodes'}")
    return paths


def read_episodes(dataset_root: Path):
    pd = load_pandas()
    frames = []
    for path in episode_parquet_paths(dataset_root):
        df = pd.read_parquet(path)
        df["__episode_parquet_path"] = str(path)
        frames.append(df)
    episodes = pd.concat(frames, ignore_index=True)
    if "episode_index" not in episodes.columns:
        raise ValueError("LeRobot episode metadata is missing `episode_index`")
    return episodes.sort_values("episode_index").reset_index(drop=True)


def video_keys_from_info(info: Dict) -> List[str]:
    keys = []
    for key, feature in info.get("features", {}).items():
        if key.startswith("observation.images.") and feature.get("dtype") == "video":
            keys.append(key)
    return keys


def camera_keys_from_info(info: Dict) -> List[str]:
    keys = []
    for key, feature in info.get("features", {}).items():
        if key.startswith("observation.images.") and feature.get("dtype") in {"image", "video"}:
            keys.append(key)
    return keys


def video_path_for_episode(dataset_root: Path, info: Dict, episode, video_key: str) -> Path:
    if f"videos/{video_key}/chunk_index" not in episode or f"videos/{video_key}/file_index" not in episode:
        raise ValueError(
            f"Episode metadata does not contain video columns for {video_key}. "
            "This dataset is likely image-backed, not video-backed. Re-export with "
            "`examples/sarm/export_rosbag_to_lerobot_sarm.py` without `--no_videos` and with `--overwrite`."
        )
    video_path_pattern = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    rel = video_path_pattern.format(
        video_key=video_key,
        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode[f"videos/{video_key}/file_index"]),
    )
    path = dataset_root / rel
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def empty_segment_annotations(subtasks: List[str]) -> List[Dict]:
    return [
        {
            "name": name,
            "start_frame": None,
            "end_frame": None,
            "start_time_sec": None,
            "end_time_sec": None,
            "notes": "",
        }
        for name in subtasks
    ]


def seconds_to_frame(seconds: float, fps: float, length: int) -> int:
    return max(0, min(length - 1, int(round(float(seconds) * fps))))


def load_existing_segments_from_columns(episode, prefix: str, fallback_names: List[str], fps: float, length: int) -> List[Dict]:
    names = episode.get(f"{prefix}_subtask_names")
    starts = episode.get(f"{prefix}_subtask_start_times")
    ends = episode.get(f"{prefix}_subtask_end_times")
    if names is None and prefix == "sparse":
        names = episode.get("subtask_names")
        starts = episode.get("subtask_start_times")
        ends = episode.get("subtask_end_times")
    if names is None or starts is None or ends is None:
        return empty_segment_annotations(fallback_names)
    try:
        if len(names) == 0:
            return empty_segment_annotations(fallback_names)
    except TypeError:
        return empty_segment_annotations(fallback_names)
    output = []
    for name, start, end in zip(list(names), list(starts), list(ends), strict=False):
        output.append(
            {
                "name": str(name),
                "start_frame": seconds_to_frame(float(start), fps, length),
                "end_frame": seconds_to_frame(float(end), fps, length),
                "start_time_sec": float(start),
                "end_time_sec": float(end),
                "notes": "",
            }
        )
    return output


def load_or_init_sidecar(path: Path, dataset_root: Path, task_desc: str, fps: float, sparse: List[str], dense: List[str]) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("schema_version", 2)
        data.setdefault("dataset_root", str(dataset_root))
        data.setdefault("task_desc", task_desc)
        data.setdefault("fps", fps)
        data.setdefault("sparse_subtasks", sparse)
        data.setdefault("dense_subtasks", dense)
        data.setdefault("episodes", [])
        return data
    return {
        "schema_version": 2,
        "format": "lerobot_sarm_manual_annotations",
        "dataset_root": str(dataset_root),
        "task_desc": task_desc,
        "fps": fps,
        "sparse_subtasks": sparse,
        "dense_subtasks": dense,
        "episodes": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def prepare_annotations(args) -> Dict:
    dataset_root = locate_dataset_root(args.dataset_root, args.repo_id, args.root)
    info = load_info(dataset_root)
    fps = float(args.fps or info.get("fps", 30))
    video_keys = video_keys_from_info(info)
    video_key = args.video_key or ("observation.images.head" if "observation.images.head" in video_keys else (video_keys[0] if video_keys else None))
    if not video_key:
        camera_keys = camera_keys_from_info(info)
        raise ValueError(
            "No video-backed observation.images.* keys found in LeRobot meta/info.json. "
            f"Camera keys found: {camera_keys}. Re-export the dataset without `--no_videos` and with `--overwrite`."
        )
    if video_key not in video_keys:
        camera_keys = camera_keys_from_info(info)
        raise ValueError(
            f"Video key {video_key!r} is not video-backed. Available video-backed keys: {video_keys}. "
            f"Camera keys found: {camera_keys}. Re-export without `--no_videos` and with `--overwrite`."
        )

    sparse = parse_csv(args.sparse_subtasks) or list(DEFAULT_SPARSE_SUBTASKS)
    dense = parse_csv(args.dense_subtasks) or list(DEFAULT_DENSE_SUBTASKS)
    annotations_path = Path(args.annotations_file).expanduser().resolve() if args.annotations_file else dataset_root / "meta" / "sarm_manual_annotations.json"
    data = load_or_init_sidecar(annotations_path, dataset_root, args.task_desc, fps, sparse, dense)
    if args.overwrite_subtasks:
        data["sparse_subtasks"] = sparse
        data["dense_subtasks"] = dense
    else:
        sparse = data.get("sparse_subtasks", sparse)
        dense = data.get("dense_subtasks", dense)

    existing_by_ep = {int(ep["episode_index"]): ep for ep in data.get("episodes", [])}
    episodes_df = read_episodes(dataset_root)
    episodes = []
    for _, row in episodes_df.iterrows():
        ep = row.to_dict()
        ep_idx = int(ep["episode_index"])
        if args.episodes and ep_idx not in args.episodes:
            continue
        length = int(ep.get("length", 0))
        video_path = video_path_for_episode(dataset_root, info, ep, video_key)
        video_start = float(ep.get(f"videos/{video_key}/from_timestamp", 0.0))
        video_end = float(ep.get(f"videos/{video_key}/to_timestamp", video_start + max(0, length - 1) / fps))
        previous = existing_by_ep.get(ep_idx)
        annotations = previous.get("annotations") if previous else None
        if annotations is None:
            annotations = {
                "sparse": load_existing_segments_from_columns(ep, "sparse", sparse, fps, length),
                "dense": load_existing_segments_from_columns(ep, "dense", dense, fps, length),
            }
        episodes.append(
            {
                "episode_index": ep_idx,
                "num_frames": length,
                "duration_sec": video_end - video_start,
                "video_path": str(video_path),
                "video_url": f"/video/{ep_idx}",
                "video_start": video_start,
                "video_end": video_end,
                "episode_parquet_path": str(ep["__episode_parquet_path"]),
                "annotations": annotations,
            }
        )

    data["dataset_root"] = str(dataset_root)
    data["fps"] = fps
    data["video_key"] = video_key
    data["available_video_keys"] = video_keys
    data["episodes"] = episodes
    data["annotations_file"] = str(annotations_path)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def valid_segments(episode: Dict, kind: str) -> List[Dict]:
    segments = []
    for item in episode.get("annotations", {}).get(kind, []):
        start = item.get("start_frame")
        end = item.get("end_frame")
        if start is None or end is None:
            continue
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        segments.append(
            {
                "name": item["name"],
                "start_frame": start,
                "end_frame": end,
                "start_time_sec": float(start) / float(episode.get("fps", 1) or 1),
                "end_time_sec": float(end) / float(episode.get("fps", 1) or 1),
            }
        )
    return segments


def compute_temporal_proportions(data: Dict, kind: str) -> Dict[str, float]:
    names = data.get(f"{kind}_subtasks", [])
    values = {name: [] for name in names}
    for episode in data.get("episodes", []):
        by_name = {}
        total = 0.0
        for item in episode.get("annotations", {}).get(kind, []):
            start = item.get("start_frame")
            end = item.get("end_frame")
            if start is None or end is None or int(end) <= int(start):
                continue
            duration = float(int(end) - int(start))
            by_name[item["name"]] = duration
            total += duration
        if total > 0:
            for name in names:
                values.setdefault(name, []).append(by_name.get(name, 0.0) / total)
    avg = {}
    for name in names:
        series = values.get(name, [])
        avg[name] = float(sum(series) / len(series)) if series else 0.0
    total = sum(avg.values())
    if total > 0:
        avg = {name: value / total for name, value in avg.items()}
    return avg


def backup_file_once(path: Path, backups: set, enabled: bool):
    if not enabled or path in backups or not path.exists():
        return
    backup = path.with_suffix(path.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    backups.add(path)


def write_annotations_to_lerobot(data: Dict, backup: bool = True, backups: Optional[set] = None):
    pd = load_pandas()
    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    fps = float(data["fps"])
    backups = backups if backups is not None else set()
    episodes_by_file: Dict[str, List[Dict]] = {}
    for episode in data.get("episodes", []):
        episode["fps"] = fps
        episodes_by_file.setdefault(episode["episode_parquet_path"], []).append(episode)

    for parquet_path_str, episodes in episodes_by_file.items():
        parquet_path = Path(parquet_path_str)
        backup_file_once(parquet_path, backups, backup)
        df = pd.read_parquet(parquet_path)
        for prefix in ("sparse", "dense"):
            cols = [
                f"{prefix}_subtask_names",
                f"{prefix}_subtask_start_times",
                f"{prefix}_subtask_end_times",
                f"{prefix}_subtask_start_frames",
                f"{prefix}_subtask_end_frames",
            ]
            for col in cols:
                if col not in df.columns:
                    df[col] = None
            if prefix == "sparse":
                for legacy in [
                    "subtask_names",
                    "subtask_start_times",
                    "subtask_end_times",
                    "subtask_start_frames",
                    "subtask_end_frames",
                ]:
                    if legacy not in df.columns:
                        df[legacy] = None

        for episode in episodes:
            ep_idx = int(episode["episode_index"])
            row_indices = df.index[df["episode_index"] == ep_idx].tolist()
            if not row_indices:
                continue
            row_idx = row_indices[0]
            for prefix in ("sparse", "dense"):
                segments = valid_segments(episode, prefix)
                names = [item["name"] for item in segments]
                starts = [float(item["start_frame"]) / fps for item in segments]
                ends = [float(item["end_frame"]) / fps for item in segments]
                start_frames = [int(item["start_frame"]) for item in segments]
                end_frames = [int(item["end_frame"]) for item in segments]
                values = [names, starts, ends, start_frames, end_frames]
                cols = [
                    f"{prefix}_subtask_names",
                    f"{prefix}_subtask_start_times",
                    f"{prefix}_subtask_end_times",
                    f"{prefix}_subtask_start_frames",
                    f"{prefix}_subtask_end_frames",
                ]
                for col, value in zip(cols, values, strict=True):
                    df.at[row_idx, col] = value
                if prefix == "sparse":
                    for col, value in zip(
                        [
                            "subtask_names",
                            "subtask_start_times",
                            "subtask_end_times",
                            "subtask_start_frames",
                            "subtask_end_frames",
                        ],
                        values,
                        strict=True,
                    ):
                        df.at[row_idx, col] = value
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    meta_dir = dataset_root / "meta"
    for kind in ("sparse", "dense"):
        proportions = compute_temporal_proportions(data, kind)
        if any(value > 0 for value in proportions.values()):
            output = meta_dir / f"temporal_proportions_{kind}.json"
            backup_file_once(output, backups, backup)
            output.write_text(json.dumps(proportions, indent=2, ensure_ascii=False), encoding="utf-8")


class AnnotationHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, annotations_path: Path, dataset_root: Path, backup: bool, **kwargs):
        self.annotations_path = annotations_path
        self.dataset_root = dataset_root
        self.backup = backup
        self.backups = kwargs.pop("backups")
        super().__init__(*args, **kwargs)

    def send_json(self, data: Dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file_range(self, path: Path, content_type: str):
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200
        if range_header:
            unit, _, range_spec = range_header.partition("=")
            if unit.strip().lower() == "bytes":
                start_str, _, end_str = range_spec.partition("-")
                if start_str.strip():
                    start = int(start_str)
                    if end_str.strip():
                        end = int(end_str)
                elif end_str.strip():
                    # Suffix range, e.g. bytes=-1048576.
                    suffix_len = int(end_str)
                    start = max(0, file_size - suffix_len)
                if start_str.strip() and end_str.strip():
                    end = int(end_str)
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers cancel in-flight range requests during seek/switch;
                    # this is expected and should not print a scary traceback.
                    return
                remaining -= len(chunk)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/annotations":
            with self.annotations_path.open("r", encoding="utf-8") as f:
                return self.send_json(json.load(f))
        if path.startswith("/video/"):
            ep_idx = int(Path(path).name)
            with self.annotations_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            episode = next((ep for ep in data.get("episodes", []) if int(ep["episode_index"]) == ep_idx), None)
            if episode is None:
                return self.send_json({"error": "episode not found"}, status=404)
            target = Path(episode["video_path"]).resolve()
            if not str(target).startswith(str(self.dataset_root.resolve())) or not target.exists():
                return self.send_json({"error": "video not found"}, status=404)
            return self.send_file_range(target, "video/mp4")
        return self.send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path != "/api/annotations":
            return self.send_json({"error": "not found"}, status=404)
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.annotations_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            write_annotations_to_lerobot(payload, backup=self.backup, backups=self.backups)
        except Exception as exc:  # noqa: BLE001
            return self.send_json({"ok": False, "error": str(exc)}, status=400)
        return self.send_json({"ok": True, "path": str(self.annotations_path)})

    def log_message(self, format, *args):  # noqa: A003
        print(f"[manual-annotate] {self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Manual SARM annotation UI for an existing LeRobotDataset.")
    parser.add_argument("--dataset_root", default=None, help="Path to an existing local LeRobotDataset root.")
    parser.add_argument("--repo_id", default=None, help="Optional LeRobot repo id. Used with --root or HF_LEROBOT_HOME lookup.")
    parser.add_argument("--root", default=None, help="Optional LeRobot root/cache path.")
    parser.add_argument("--annotations_file", default=None, help="Annotation sidecar JSON. Defaults to <dataset_root>/meta/sarm_manual_annotations.json.")
    parser.add_argument("--task_desc", default=DEFAULT_TASK_DESC)
    parser.add_argument("--fps", type=float, default=None, help="Override dataset fps if needed.")
    parser.add_argument("--video_key", default="observation.images.head")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode indices to annotate.")
    parser.add_argument("--sparse_subtasks", default=",".join(DEFAULT_SPARSE_SUBTASKS))
    parser.add_argument("--dense_subtasks", default=",".join(DEFAULT_DENSE_SUBTASKS))
    parser.add_argument("--overwrite_subtasks", action="store_true")
    parser.add_argument("--no_backup", action="store_true", help="Do not backup LeRobot parquet/proportion files before first write.")
    parser.add_argument("--prepare_only", action="store_true", help="Write sidecar JSON and exit without starting the web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    data = prepare_annotations(args)
    annotations_path = Path(data["annotations_file"]).expanduser().resolve()
    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    print(f"LeRobot dataset root: {dataset_root}")
    print(f"Annotation JSON: {annotations_path}")
    print(f"Video key: {data['video_key']}")
    print(f"Episodes loaded: {len(data['episodes'])}")
    if args.prepare_only:
        return

    backups = set()
    handler = partial(
        AnnotationHandler,
        annotations_path=annotations_path,
        dataset_root=dataset_root,
        backup=not args.no_backup,
        backups=backups,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Manual annotation UI: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
