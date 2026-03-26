import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests


def decode_image_base64(data: Optional[str]) -> Optional[np.ndarray]:
    if not data:
        return None
    raw = base64.b64decode(data)
    np_arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img[..., ::-1]


class R1LiteClient:
    def __init__(self, server_url: str, timeout: float = 2.0):
        self.server_url = server_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def _url(self, path: str) -> str:
        return self.server_url + path.lstrip("/")

    def get_state(self) -> Dict[str, Any]:
        response = self.session.get(self._url("state"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_health(self) -> Dict[str, Any]:
        response = self.session.get(self._url("health"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def post_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(self._url("action"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def set_mode(self, mode: str, owner: str = "debug") -> Dict[str, Any]:
        response = self.session.post(self._url("mode"), json={"mode": mode, "owner": owner}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def hold(self, owner: str = "debug") -> Dict[str, Any]:
        response = self.session.post(self._url("hold"), json={"mode": "hold", "owner": owner}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def recover(self, owner: str = "debug") -> Dict[str, Any]:
        response = self.session.post(self._url("recover"), json={"mode": "recover", "owner": owner}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def reset(self, left_pose=None, right_pose=None, torso=None, owner: str = "debug") -> Dict[str, Any]:
        payload = {"owner": owner}
        if left_pose is not None:
            payload["left_pose"] = left_pose
        if right_pose is not None:
            payload["right_pose"] = right_pose
        if torso is not None:
            payload["torso"] = torso
        response = self.session.post(self._url("reset"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
