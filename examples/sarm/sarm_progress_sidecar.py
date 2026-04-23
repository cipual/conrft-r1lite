#!/usr/bin/env python3
"""HTTP sidecar for online SARM progress inference.

The RL environment posts the current camera image and state to /predict_progress.
This sidecar runs in the `lerobot` environment, loads a trained SARM model, and
returns a scalar progress value in [0, 1].
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from lerobot.policies.sarm.modeling_sarm import SARMRewardModel


LOG = logging.getLogger("sarm-progress-sidecar")


def _clip_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if hasattr(output, "to_tuple"):
        for value in output.to_tuple():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")


def _decode_image(image_jpeg_base64: str) -> Image.Image:
    raw = base64.b64decode(image_jpeg_base64)
    image = Image.open(BytesIO(raw)).convert("RGB")
    return image


class SARMProgressPredictor:
    def __init__(self, reward_model_path: str, device: str, default_head_mode: str):
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.default_head_mode = default_head_mode

        LOG.info("Loading SARM model from %s", reward_model_path)
        self.reward_model = SARMRewardModel.from_pretrained(reward_model_path)
        self.reward_model.config.device = str(self.device)
        self.reward_model.to(self.device).eval()

        LOG.info("Loading CLIP encoder")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device).eval()

        # Online inference only has the current observation. The SARM model was
        # trained with a temporal window, so we repeat the current frame/state to
        # fill the observation context and query the current-frame slot.
        self.num_context_frames = int(self.reward_model.config.n_obs_steps) + 1
        self.target_frame_index = int(self.reward_model.config.n_obs_steps)
        LOG.info(
            "SARM sidecar ready: device=%s context_frames=%d target_frame_index=%d default_head_mode=%s",
            self.device,
            self.num_context_frames,
            self.target_frame_index,
            self.default_head_mode,
        )

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        return _clip_output_to_tensor(self.clip_model.get_image_features(**inputs)).float()

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        inputs = self.clip_processor(text=[text or ""], return_tensors="pt", padding=True, truncation=True).to(self.device)
        return _clip_output_to_tensor(self.clip_model.get_text_features(**inputs)).float()

    @torch.no_grad()
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image = _decode_image(payload["image_jpeg_base64"])
        task = str(payload.get("task", ""))
        head_mode = str(payload.get("head_mode") or self.default_head_mode)
        state = np.asarray(payload.get("state", []), dtype=np.float32).reshape(-1)
        if state.size == 0:
            state = np.zeros((1,), dtype=np.float32)

        image_features = self._encode_image(image)
        text_features = self._encode_text(task)

        video_features = image_features.unsqueeze(1).repeat(1, self.num_context_frames, 1)
        state_features = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        state_features = state_features.repeat(1, self.num_context_frames, 1)
        lengths = torch.full((1,), self.num_context_frames, dtype=torch.int32, device=self.device)

        progress, stage_probs = self.reward_model.calculate_rewards(
            text_embeddings=text_features,
            video_embeddings=video_features,
            state_features=state_features,
            lengths=lengths,
            return_stages=True,
            head_mode=head_mode,
            frame_index=self.target_frame_index,
        )
        progress_value = float(np.clip(np.asarray(progress).reshape(-1)[0], 0.0, 1.0))
        stage_probs_arr = np.asarray(stage_probs)
        if stage_probs_arr.ndim == 3:
            stage_probs_arr = stage_probs_arr[0, self.target_frame_index]
        elif stage_probs_arr.ndim == 2:
            stage_probs_arr = stage_probs_arr[self.target_frame_index]
        stage_index = int(np.argmax(stage_probs_arr)) if stage_probs_arr.size else -1
        confidence = float(np.max(stage_probs_arr)) if stage_probs_arr.size else 0.0
        return {
            "progress": progress_value,
            "head_mode": head_mode,
            "stage_index": stage_index,
            "stage_confidence": confidence,
        }


class Handler(BaseHTTPRequestHandler):
    predictor: SARMProgressPredictor

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/predict_progress":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.predictor.predict(payload)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Prediction failed")
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Tuple[Any, ...]) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward_model_path", required=True, help="Path to the trained SARM pretrained_model directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default_head_mode", choices=("sparse", "dense"), default="dense")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    Handler.predictor = SARMProgressPredictor(
        reward_model_path=args.reward_model_path,
        device=args.device,
        default_head_mode=args.default_head_mode,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOG.info("Serving SARM progress endpoint at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
